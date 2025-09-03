import json
import math
import os
import queue
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from transformers import LogitsProcessor


def barycenter_reverse_kl_weights(P, tol=1e-8, max_iter=200, eps=1e-12):
    """
    Minimize sum_i KL(P_i || P*) with P* = sum_i w_i P_i, w in simplex.
    Args:
        P: tensor of shape [K, V], each row a distribution (nonneg, sum=1).
        tol: L1 tol on w change.
        max_iter: max iterations.
        eps: small smoothing to avoid division by zero.
    Returns:
        w: [K] optimal weights (sum=1, >=0)
        P_star: [V] barycenter
    """

    K, V = P.shape
    # normalize defensively
    P = P.clamp_min(0)
    P = P / (P.sum(dim=1, keepdim=True) + 1e-20)

    Q = P.sum(dim=0)                      # [V]
    w = torch.full((K,), 1.0 / K, dtype=P.dtype, device=P.device)

    for _ in range(max_iter):
        P_star = (w[:, None] * P).sum(dim=0)           # [V]
        # smoothing for stability
        P_star = (1 - eps) * P_star + eps / V

        # s_k = sum_x Q(x) * P_k(x) / P_star(x)
        ratio = Q / P_star                             # [V]
        s = (P * ratio[None, :]).sum(dim=1)            # [K]

        w_new_unnorm = w * s
        w_new_sum = w_new_unnorm.sum()
        # if degenerate (can happen early), fall back to uniform
        if w_new_sum <= 0:
            w_new = torch.full_like(w, 1.0 / K)
        else:
            w_new = w_new_unnorm / w_new_sum

        if torch.sum(torch.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    P_star = (w[:, None] * P).sum(dim=0)
    P_star = P_star / P_star.sum()
    return w, P_star


class BasedOnProbabilityTransferLogits_Loacal_FP32_Processor(LogitsProcessor):
    def __init__(self, learning_rate, learning_epochs_nums, ensemble_weight,
                 ensemble_model_output_ids_queue, assist_model_score_queue_list,
                 main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, result_save_dir, main_model_tokenizer,
                 assist_model_tokenizer, device, device_compute, ensemble_method, early_stop_string_list=None):
        self.learning_rate = learning_rate
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.learning_epochs_nums = learning_epochs_nums
        self.ensemble_weight = ensemble_weight
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.result_save_dir = result_save_dir
        self.main_model_tokenizer = main_model_tokenizer
        self.assist_model_tokenizer_list = assist_model_tokenizer
        self.device = device
        self.device_compute = device_compute
        self.ensemble_method = ensemble_method
        self.early_stop_string_list = early_stop_string_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ensemble_process_file_path = os.path.join(self.result_save_dir,
                                                  f'ensemble_lr{self.learning_rate}_anchor_point_count_all_learning_epochs_nums_5.log')
        main_model_only_flag = False
        json_object = {}

        assist_model_generate_ids_logits_list = []
        for index, queue_instance in enumerate(self.assist_model_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=5)
                assist_model_generate_ids_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_model_generate_ids_logits_list.append(None)
                main_model_only_flag = True

        if len(assist_model_generate_ids_logits_list) == 0:
            main_model_only_flag = True
        if math.fabs(self.learning_rate) <= 1e-6:
            main_model_only_flag = True
        if torch.argmax(scores).item() == self.main_model_tokenizer.eos_token_id:
            main_model_only_flag = True

        if self.early_stop_string_list is not None:
            for early_stop_string in self.early_stop_string_list:
                early_stop_token = self.main_model_tokenizer(early_stop_string, return_tensors="pt",
                                                             add_special_tokens=False).input_ids.tolist()[0][1:]
                last_token_count = len(early_stop_token)

                last_token_ids = input_ids.tolist()[0][-last_token_count:]
                if last_token_ids == early_stop_token:
                    scores[:, self.main_model_tokenizer.eos_token_id] = float('inf')
                    main_model_only_flag = True

        if not main_model_only_flag:

            main_model_generate_ids_logits = Variable(scores, requires_grad=True).to(torch.float32).to(
                self.main_model_probability_transfer_matrix_list[0].device)

            with torch.no_grad():
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits,
                                                                      dim=-1)

                main_model_generate_ids_probs_values, main_model_generate_ids_probs_indices = torch.topk(
                    main_model_generate_ids_probs, k=10)
                json_object[f'origin_main_top_tokens'] = self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_probs_indices.tolist()[0])

                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[
                                                                        0]).to(self.device_compute)
                # Print the shape of main_model_relative_representation_probs
                # print(f"main_model_relative_representation_probs dim:{main_model_relative_representation_probs.shape}")

                main_model_relative_values, main_model_relative_indices = torch.topk(
                    main_model_relative_representation_probs, k=10)
                json_object[f'main_rel_values'] = main_model_relative_values.tolist()[0]
                json_object[f'main_rel_indices'] = main_model_relative_indices.tolist()[0]

                model_relative_representation_probs_list = [main_model_relative_representation_probs]

                for index, (assist_model_generate_ids_logits, assist_model_probability_transfer_matrix) in enumerate(
                        zip(assist_model_generate_ids_logits_list,
                            self.assist_model_probability_transfer_matrix_list)):
                    assist_model_generate_ids_probs = nn.functional.softmax(
                        assist_model_generate_ids_logits.to(torch.float32),
                        dim=-1).to(assist_model_probability_transfer_matrix.device)

                    values, indices = torch.topk(assist_model_generate_ids_probs, k=10)
                    json_object[f'origin_aux_{index}_top_tokens'] = self.assist_model_tokenizer_list[
                        index].convert_ids_to_tokens(indices.tolist()[0])

                    assist_model_relative_representation_probs = torch.mm(assist_model_generate_ids_probs,
                                                                          assist_model_probability_transfer_matrix).to(
                        self.device_compute)
                    # Print the shape of assist_model_relative_representation_probs
                    # print(f"assist_model_relative_representation_probs dim:{assist_model_relative_representation_probs.shape}")

                    assist_model_relative_values, assist_model_relative_indices = torch.topk(
                        assist_model_relative_representation_probs, k=10)
                    json_object[f'aux_rel_values_{index}'] = assist_model_relative_values.tolist()[0]
                    json_object[f'aux_rel_indices_{index}'] = assist_model_relative_indices.tolist()[0]

                    model_relative_representation_probs_list.append(assist_model_relative_representation_probs)

            json_object[f'self.ensemble_weight'] = self.ensemble_weight
            # print(self.ensemble_weight)
            average_probs = torch.zeros_like(main_model_relative_representation_probs)

            # --- start of ensemble ---

            model_relative_representation_probs_mat = torch.stack([probs.flatten() for probs in model_relative_representation_probs_list], dim=0)
            weight_vec = torch.tensor(self.ensemble_weight, device=model_relative_representation_probs_mat.device).unsqueeze(1)

            p_star = main_model_relative_representation_probs
            if self.ensemble_method[:4] == "tas2":
                p_star = torch.mean(model_relative_representation_probs_mat, dim=0, keepdim=True)

            token_conf_mat = torch.ones_like(model_relative_representation_probs_mat)
            if self.ensemble_method != "vanilla":
                token_conf_mat = torch.exp(-torch.abs(model_relative_representation_probs_mat - p_star))

            average_probs = torch.sum(weight_vec * token_conf_mat * model_relative_representation_probs_mat, dim=0, keepdim=True)

            if self.ensemble_method == "tas+mas":
                agreed_probs_tensor = F.normalize(weight_vec * token_conf_mat * model_relative_representation_probs_mat, p=1, dim=1)
                _, average_probs = barycenter_reverse_kl_weights(agreed_probs_tensor)
                average_probs = average_probs.unsqueeze(0)
            elif self.ensemble_method[-4:] == "mas2":
                model_conf_vec = token_conf_mat.sum(dim=1, keepdim=True)
                average_probs = torch.sum(weight_vec * model_conf_vec * token_conf_mat * model_relative_representation_probs_mat, dim=0, keepdim=True)

            average_probs = F.normalize(average_probs, p=1, dim=-1)

            # --- end of ensemble ---
            
            average_relative_probs_values, average_relative_probs_indices = torch.topk(average_probs, k=10)

            json_object[f'average_rel_probs_values'] = average_relative_probs_values.tolist()[0]
            json_object[f'average_rel_probs_indices'] = average_relative_probs_indices.tolist()[0]

            torch.set_grad_enabled(True)
            main_model_generate_ids_logits = main_model_generate_ids_logits.to(self.device_compute).detach().clone().to(
                torch.float32)
            main_model_generate_ids_logits.requires_grad_(True)
            local_main_model_relative_representation_matrix = self.main_model_probability_transfer_matrix_list[0].to(
                self.device_compute)
            local_learning_rate = self.learning_rate
            criterion = nn.KLDivLoss()

            optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
                                          lr=local_learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=local_learning_rate / 4)

            for i in range(0, self.learning_epochs_nums):
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    local_main_model_relative_representation_matrix)

                log_main_probs = torch.log(main_model_relative_representation_probs)
                loss = criterion(log_main_probs, average_probs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                    torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
                json_object[f'main_model_generate_ids_logits_probs_values_{i}'] = \
                    main_model_generate_ids_logits_probs_values.tolist()[0]
                json_object[f'main_model_generate_ids_logits_indices_{i}'] = \
                    self.main_model_tokenizer.convert_ids_to_tokens(
                        main_model_generate_ids_logits_indices.tolist()[0])

            torch.set_grad_enabled(False)

            next_tokens_id = torch.argmax(main_model_generate_ids_logits, dim=-1)

            main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
            json_object[f'main_model_generate_ids_logits_probs_values_final'] = \
                main_model_generate_ids_logits_probs_values.tolist()[0]
            json_object[f'main_model_generate_ids_logits_indices_final'] = \
                self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_logits_indices.tolist()[0])

            self.ensemble_model_output_ids_queue.put(next_tokens_id)
            with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            return main_model_generate_ids_logits.to(self.device).detach()

        else:
            next_tokens_id = torch.argmax(scores, dim=-1)
            self.ensemble_model_output_ids_queue.put(next_tokens_id)
            return scores


class BasedOnProbabilityTransferLogits_Loacal_FP32_digit_vote_Processor(LogitsProcessor):
    def __init__(self, learning_rate, learning_epochs_nums, ensemble_weight,
                 ensemble_model_output_ids_queue, assist_model_score_queue_list,
                 main_model_probability_transfer_matrix_list,
                 assist_model_probability_transfer_matrix_list, result_save_dir, main_model_tokenizer,
                 assist_model_tokenizer, device, device_compute, early_stop_string_list=None):
        self.learning_rate = learning_rate
        self.assist_model_score_queue_list = assist_model_score_queue_list
        self.learning_epochs_nums = learning_epochs_nums
        self.ensemble_weight = ensemble_weight
        self.ensemble_model_output_ids_queue = ensemble_model_output_ids_queue
        self.main_model_probability_transfer_matrix_list = main_model_probability_transfer_matrix_list
        self.assist_model_probability_transfer_matrix_list = assist_model_probability_transfer_matrix_list
        self.result_save_dir = result_save_dir
        self.main_model_tokenizer = main_model_tokenizer
        self.assist_model_tokenizer_list = assist_model_tokenizer
        self.device = device
        self.device_compute = device_compute
        self.early_stop_string_list = early_stop_string_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ensemble_process_file_path = os.path.join(self.result_save_dir,
                                                  f'ensemble_lr{self.learning_rate}_anchor_point_count_all_learning_epochs_nums_5.log')
        main_model_only_flag = False
        json_object = {}

        assist_model_generate_ids_logits_list = []
        for index, queue_instance in enumerate(self.assist_model_score_queue_list):
            try:
                value = queue_instance.get(block=True, timeout=5)
                assist_model_generate_ids_logits_list.append(value)

            except queue.Empty:
                print(f"aux model{index}【not received】\n")
                assist_model_generate_ids_logits_list.append(None)
                main_model_only_flag = True

        if len(assist_model_generate_ids_logits_list) == 0:
            main_model_only_flag = True
        if math.fabs(self.learning_rate) <= 1e-6:
            main_model_only_flag = True
        if torch.argmax(scores).item() == self.main_model_tokenizer.eos_token_id:
            main_model_only_flag = True

        if self.early_stop_string_list is not None:
            for early_stop_string in self.early_stop_string_list:
                early_stop_token = self.main_model_tokenizer(early_stop_string, return_tensors="pt",
                                                             add_special_tokens=False).input_ids.tolist()[0][1:]
                last_token_count = len(early_stop_token)

                last_token_ids = input_ids.tolist()[0][-last_token_count:]
                if last_token_ids == early_stop_token:
                    scores[:, self.main_model_tokenizer.eos_token_id] = float('inf')
                    main_model_only_flag = True

        if self.main_model_tokenizer.decode(torch.argmax(scores).item()).isdigit():

            candidates_list = [self.main_model_tokenizer.decode(torch.argmax(scores).item())]
            for index, logits in enumerate(assist_model_generate_ids_logits_list):
                output = self.assist_model_tokenizer_list[index].decode(torch.argmax(logits).item())
                # if output.isdigit():
                candidates_list.append(output)
            counter = Counter(candidates_list)

            print(f"本轮数字投票集成:{candidates_list}")
            most_common_number = counter.most_common(1)[0][0]
            if most_common_number.isdigit():
                vote_result_token_id = self.main_model_tokenizer.convert_tokens_to_ids(f"{most_common_number}")
                self.ensemble_model_output_ids_queue.put(torch.tensor([vote_result_token_id]).to(self.device))
                json_object[f'digit_vote'] = candidates_list
                json_object[f'digit_vote_result'] = most_common_number
                with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                    process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

                output = torch.zeros_like(scores).to(self.device)
                output[:, vote_result_token_id] = float('inf')
                return output

        if not main_model_only_flag:

            main_model_generate_ids_logits = Variable(scores, requires_grad=True).to(torch.float32).to(
                self.main_model_probability_transfer_matrix_list[0].device)

            with torch.no_grad():
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits,
                                                                      dim=-1)

                main_model_generate_ids_probs_values, main_model_generate_ids_probs_indices = torch.topk(
                    main_model_generate_ids_probs, k=10)
                json_object[f'origin_main_top_tokens'] = self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_probs_indices.tolist()[0])

                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.main_model_probability_transfer_matrix_list[
                                                                        0]).to(self.device_compute)

                main_model_relative_values, main_model_relative_indices = torch.topk(
                    main_model_relative_representation_probs, k=10)
                json_object[f'main_rel_values'] = main_model_relative_values.tolist()[0]
                json_object[f'main_rel_indices'] = main_model_relative_indices.tolist()[0]

                model_relative_representation_probs_list = [main_model_relative_representation_probs]

                for index, (assist_model_generate_ids_logits, assist_model_probability_transfer_matrix) in enumerate(
                        zip(assist_model_generate_ids_logits_list,
                            self.assist_model_probability_transfer_matrix_list)):
                    assist_model_generate_ids_probs = nn.functional.softmax(
                        assist_model_generate_ids_logits.to(torch.float32),
                        dim=-1).to(assist_model_probability_transfer_matrix.device)

                    values, indices = torch.topk(assist_model_generate_ids_probs, k=10)
                    json_object[f'origin_aux_{index}_top_tokens'] = self.assist_model_tokenizer_list[
                        index].convert_ids_to_tokens(indices.tolist()[0])

                    assist_model_relative_representation_probs = torch.mm(assist_model_generate_ids_probs,
                                                                          assist_model_probability_transfer_matrix).to(
                        self.device_compute)

                    assist_model_relative_values, assist_model_relative_indices = torch.topk(
                        assist_model_relative_representation_probs, k=10)
                    json_object[f'aux_rel_values_{index}'] = assist_model_relative_values.tolist()[0]
                    json_object[f'aux_rel_indices_{index}'] = assist_model_relative_indices.tolist()[0]

                    model_relative_representation_probs_list.append(assist_model_relative_representation_probs)

            json_object[f'ensemble_weight'] = self.ensemble_weight

            average_probs = torch.zeros_like(main_model_relative_representation_probs)
            for weight, probs in zip(self.ensemble_weight, model_relative_representation_probs_list):
                average_probs += weight * probs

            average_relative_probs_values, average_relative_probs_indices = torch.topk(
                average_probs, k=10)

            json_object[f'average_rel_probs_values'] = average_relative_probs_values.tolist()[0]
            json_object[f'average_rel_probs_indices'] = average_relative_probs_indices.tolist()[0]

            torch.set_grad_enabled(True)
            main_model_generate_ids_logits = main_model_generate_ids_logits.to(self.device_compute).detach().clone().to(
                torch.float32)
            main_model_generate_ids_logits.requires_grad_(True)
            local_main_model_relative_representation_matrix = self.main_model_probability_transfer_matrix_list[0].to(
                self.device_compute)
            local_learning_rate = self.learning_rate
            criterion = nn.KLDivLoss()

            optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
                                          lr=local_learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=local_learning_rate / 4)

            for i in range(0, self.learning_epochs_nums):
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    local_main_model_relative_representation_matrix)

                log_main_probs = torch.log(main_model_relative_representation_probs)
                loss = criterion(log_main_probs, average_probs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                    torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
                json_object[f'main_model_generate_ids_logits_probs_values_{i}'] = \
                    main_model_generate_ids_logits_probs_values.tolist()[0]
                json_object[f'main_model_generate_ids_logits_indices_{i}'] = \
                    self.main_model_tokenizer.convert_ids_to_tokens(
                        main_model_generate_ids_logits_indices.tolist()[0])

            torch.set_grad_enabled(False)

            next_tokens_id = torch.argmax(main_model_generate_ids_logits, dim=-1)

            main_model_generate_ids_logits_probs_values, main_model_generate_ids_logits_indices = torch.topk(
                torch.nn.functional.softmax(main_model_generate_ids_logits, dim=-1), k=10)
            json_object[f'main_model_generate_ids_logits_probs_values_final'] = \
                main_model_generate_ids_logits_probs_values.tolist()[0]
            json_object[f'main_model_generate_ids_logits_indices_final'] = \
                self.main_model_tokenizer.convert_ids_to_tokens(
                    main_model_generate_ids_logits_indices.tolist()[0])

            self.ensemble_model_output_ids_queue.put(next_tokens_id)
            with open(ensemble_process_file_path, "a+", encoding="utf-8") as process_file:
                process_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')

            return main_model_generate_ids_logits.to(self.device).detach()

        else:
            next_tokens_id = torch.argmax(scores, dim=-1)
            self.ensemble_model_output_ids_queue.put(next_tokens_id)
            return scores
