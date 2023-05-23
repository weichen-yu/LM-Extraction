# Copyright 2022 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
from absl import logging
import csv
import os
import numba as nb
import tempfile
from typing import Tuple, Union

import numpy as np
import transformers
import torch
import time
import torch.backends.cudnn as cudnn
import random
_ROOT_DIR = flags.DEFINE_string(
    'root-dir', "tmp/",
    "Path to where (even intermediate) results should be saved/loaded."
)
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment-name',
    'sample',
    "Name of the experiment. This defines the subdir in `root_dir` where "
    "results are saved.")
_DATASET_DIR = flags.DEFINE_string(
    "dataset-dir", "../datasets",
    "Path to where the data lives.")
_DATSET_FILE = flags.DEFINE_string(
    "dataset-file", "train_dataset.npy", "Name of dataset file to load.")
_NUM_TRIALS = flags.DEFINE_integer(
    'num-trials', 5, 'Number of generations per prompt.')
_local_rank = flags.DEFINE_integer(
    'local_rank', 0, 'cuda num')
_generation_exists = flags.DEFINE_integer(
    'generation_exists', 0, 'if 0, overwrite previous ones')
_val_set_num = flags.DEFINE_integer(
    'val_set_num', 1000, 'test set')
_seed = flags.DEFINE_integer(
    'seed', 2022, 'random seed')
_SUFFIX_LEN = 50
_PREFIX_LEN = 50
_MODEL = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
_MODEL = _MODEL.half().cuda().eval()

# = _local_rank
# torch.distributed.init_process_group('nccl', init_method='env://')
# if torch.distributed.get_world_size() != torch.cuda.device_count():
#     raise AssertionError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
#         torch.distributed.get_world_size(), torch.cuda.device_count()))

def init_seeds(_seed):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    return

@nb.jit()
def pick_values(all_losses, all_generations, indexes):
    pick_likelihood = []
    pick_generations = []
    for i in range(all_losses.shape[0]):
        pick_likelihood.append(all_losses[i][indexes[i]])
        pick_generations.append(all_generations[i][:, indexes[i]])
    return pick_likelihood, pick_generations

import numpy as np

def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    """
    多维数组排序
    Args:
        array: 多维数组
        top_k: 取数
        axis: 轴维度
        reverse: 是否倒序

    Returns:
        top_sorted_scores: 值
        top_sorted_indexes: 位置
    """
    if reverse:
        # argpartition分区排序，在给定轴上找到最小的值对应的idx，partition同理找对应的值
        # kth表示在前的较小值的个数，带来的问题是排序后的结果两个分区间是仍然是无序的
        # kth绝对值越小，分区排序效果越明显
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    # 分区后重新排序
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes


def generate_for_prompts(
    prompts: np.ndarray, batch_size: int=32) -> Tuple[np.ndarray, np.ndarray]:
    """Generates suffixes given `prompts` and scores using their likelihood.

    Args:
    prompts: A np.ndarray of shape [num_prompts, prefix_length]. These
        provide the context for generating each suffix. Each value should be an
        int representing the token_id. These are directly provided by loading the
        saved datasets from extract_dataset.py.
    batch_size: The number of prefixes to generate suffixes for
        sequentially.

    Returns:
        A tuple of generations and their corresponding likelihoods.
        The generations take shape [num_prompts, _SUFFIX_LEN].
        The likelihoods take shape [num_prompts]
    """
    generations = []
    losses = []
    generation_len = _SUFFIX_LEN + _PREFIX_LEN
    batch_size = 32

    for i, off in enumerate(range(0, len(prompts), batch_size)):
        prompt_batch = prompts[off:off+batch_size]
        prompt_batch = np.stack(prompt_batch, axis=0)
        input_ids = torch.tensor(prompt_batch, dtype=torch.int64)

        with torch.no_grad():
            # 1. Generate outputs from the model
            generated_tokens = _MODEL.generate(
                input_ids.cuda(),
                max_length=generation_len,  #100
                do_sample=True, 
                top_k=10,
                # temperature=0.3,
                # num_beams=2,
                # pos_loc_len=0,
                # top_p=0.85,
                # typical_p=0.7,
                # repetition_penalty=0.92,
                pad_token_id=50256  # Silences warning.
            ).cpu().detach()

            # 2. Compute each sequence's probability, excluding EOS and SOS.
            outputs = _MODEL(
                generated_tokens.cuda(),
                labels=generated_tokens.cuda(),
            )

            logits = outputs.logits.cpu().detach() #50(batch)*100*50257
            logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float() #4950(50*99)*50257
            topk = 2
            top_sorted_scores, top_sorted_indexes = logits.topk(topk, dim=-1)

            flag1 = (top_sorted_scores[:, 0] - top_sorted_scores[:, 1]) > 0.5
            flag2 = top_sorted_scores[:, 0] > 0
            flag = flag1 * flag2    

            # logits = (logits - torch.min(logits)) / torch.max(logits)

            loss_per_token = torch.nn.functional.cross_entropy(
                logits, generated_tokens[:, 1:].flatten(), reduction='none') #4950

            loss_per_token_np = loss_per_token.numpy()
            mean = np.mean(loss_per_token_np, axis=0)
            std = np.std(loss_per_token_np, axis=0)

            floor = mean - 3*std
            upper = mean + 3*std
            # # import pdb; pdb.set_trace()

            for i, val in enumerate(loss_per_token_np):
                # loss_per_token_np[i] = float(np.where(((val<floor)|(val>upper)), mean, val))
                loss_per_token_np[i] = float(np.where(((val<floor)|(val>upper)), 1., val))

            # loss_per_token = torch.tensor(loss_per_token_np)
            #loss_per_token = loss_per_token - flag2 * 0.05
            loss_per_token = loss_per_token - (flag1.int() - flag2.int()) * mean * 0.15
            # loss_per_token = torch.sigmoid(loss_per_token) + flag * 0.1
            # import pdb; pdb.set_trace()
            loss_per_token = loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN:] #50(batchsize)*50(length)
            # import pdb; pdb.set_trace()

            # print(loss_per_token_np)

            # likelihood = torch.tensor(loss_per_token_np).mean()

            likelihood = loss_per_token.mean(1) 
            #likelihood = loss_per_token_np.mean(1) 
            
            generations.extend(generated_tokens.numpy())
            losses.extend(likelihood.numpy())
    return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1))

def write_array(
    file_path: str, array: np.ndarray, unique_id: Union[int, str]):
    """Writes a batch of `generations` and `losses` to a file.

    Formats a `file_path` (e.g., "/tmp/run1/batch_{}.npy") using the `unique_id`
    so that each batch goes to a separate file. This function can be used in
    multiprocessing to speed this up.

    Args:
        file_path: A path that can be formatted with `unique_id`
        array: A numpy array to save.
        unique_id: A str or int to be formatted into `file_path`. If `file_path`
          and `unique_id` are the same, the files will collide and the contents
          will be overwritten.
    """
    file_ = file_path.format(unique_id)
    np.save(file_, array)

def hamming(gt, generate):
    if len(generate.shape) == 2:
        hamming_dist = (gt != generate).sum(1)
    else:
        hamming_dist = (gt != generate[0]).sum(1)
    return hamming_dist.mean(), hamming_dist.shape

def gt_position(answers,batch_size=50):
    gt_loss = []
    for i, off in enumerate(range(0, len(answers), batch_size)):
        answers_batch = answers[off:off+batch_size]
        answers_batch = np.stack(answers_batch, axis=0)
        with torch.no_grad():
            outputs = _MODEL(
                answers.cuda(),
                labels=answers.cuda(),
            )
            answers_logits = outputs.logits.cpu().detach()
            answers_logits = answers_logits[:, :-1].reshape((-1, answers_logits.shape[-1])).float()
            answers_loss_per_token = torch.nn.functional.cross_entropy(
                answers_logits, answers[:, 1:].flatten(), reduction='none')
            answers_loss_per_token = answers_loss_per_token.reshape((-1, generation_len - 1))[:,-_SUFFIX_LEN-1:-1]
            likelihood = answers_loss_per_token.mean(1)
            
            gt_loss.extend(likelihood.numpy())
    return gt_loss

def compare_loss(gt_loss,gene_loss):
    loss_all = np.concatenate((gt_loss,gene_loss),axis=1)
    loss_ranked = np.sort(loss_all,axis=1)
    argrank = np.argsort(loss_all,axis=1)
    top1 = argrank()
    return loss_ranked,argrank,top1,top5

def plot_hist(loss):
    return

def load_prompts(dir_: str, file_name: str) -> np.ndarray:
    """Loads prompts from the file pointed to `dir_` and `file_name`."""
    return np.load(os.path.join(dir_, file_name)).astype(np.int64)

def is_memorization(guesses, answers):
    # precision = 0
    # for guess in guesses:
    #     precision += min(np.sum(np.all(guess == answers, axis=-1)),1)
    # precision = precision/guesses.shape[0]
    # return precision
    return  np.sum(np.all(guesses==answers, axis=-1))/guesses.shape[0]

def error_100(guesses_order, order, answers):
    error=0
    i=0
    while error <= 100:
        if np.all(guesses_order[i]==answers[order[i]],axis=-1):
            i += 1
        else:
            error += 1
            i += 1
    return i, i-100

def precision_multiprompts(generations, answers, num_perprompt):
    precision_multi = 0
    generations = generations[:,:num_perprompt,:]
    for i in range(generations.shape[0]):
        # is_in = 0
        # for prompt in generation:
        #     is_in += min(np.sum(np.all(prompt == answers, axis=-1)),1)
        # precision_multi += min(is_in,1)
        if np.sum(np.all(generations[i] == answers[i],axis=-1)):
            precision_multi += 1
    precision_multi = precision_multi/generations.shape[0]
    return precision_multi

def prepare_data(val_set_num):
    experiment_base = os.path.join(_ROOT_DIR.value, _EXPERIMENT_NAME.value)
    generations_base = os.path.join(experiment_base, "generations")
    os.makedirs(generations_base, exist_ok=True)
    losses_base = os.path.join(experiment_base, "losses")
    os.makedirs(losses_base, exist_ok=True)
    prompts = load_prompts(_DATASET_DIR.value, "train_prefix.npy")[-val_set_num:]
    return experiment_base, generations_base, losses_base, prompts

def write_guesses_order(generations_per_prompt, order, guesses_order):
    with open("guess%d.csv"%generations_per_prompt, "w") as file_handle:
        print("Writing out guess with", generations_per_prompt)
        writer = csv.writer(file_handle)
        writer.writerow(["Example ID", "Suffix Guess"])

        # Write out the guesses
        for example_id, guess in zip(order, guesses_order):
            row_output = [
                example_id, str(list(guess)).replace(" ", "")
            ]
            writer.writerow(row_output)
    return

def metric_print(generations_one, all_generations, generations_per_prompt, generations_order, order, val_set_num):
    answers = np.load(os.path.join(_DATASET_DIR.value, "train_dataset.npy"))[-val_set_num:,-100:].astype(np.int64) #15000, 100
    print('generations and answer shape:', all_generations.shape, answers.shape)
    precision = is_memorization(generations_one, answers)
    print('precision:', precision)
    percision_multi = precision_multiprompts(all_generations, answers,generations_per_prompt)
    print('precision_multi:', percision_multi)
    error_k = error_100(generations_order, order, answers)
    print('error100 number:', error_k)
    ham_dist = hamming(answers, generations_one)
    print('hamming dist:', ham_dist)
    return precision, percision_multi, error_k, ham_dist

def main(_):
    init_seeds(_seed.value)
    start_t = time.time()
    
    experiment_base, generations_base, losses_base, prompts = prepare_data(_val_set_num.value)
    all_generations, all_losses = [], []
    #if not all([os.listdir(generations_base), os.listdir(losses_base)]):
    if not _generation_exists.value:
        for trial in range(_NUM_TRIALS.value):
            print('trial:',trial)
            start_t = time.time()
            os.makedirs(experiment_base, exist_ok=True)
            generations, losses = generate_for_prompts(prompts)

            generation_string = os.path.join(generations_base, "{}.npy")
            losses_string = os.path.join(losses_base, "{}.npy")
            write_array(generation_string, generations, trial)
            write_array(losses_string, losses, trial)

            all_generations.append(generations)
            all_losses.append(losses)
        all_generations = np.stack(all_generations, axis=1)
        all_losses = np.stack(all_losses, axis=1)
        print('time_consumed:',time.time()-start_t)

    else:  # Load saved results because we did not regenerate them.
        all_generations = []
        for generation_file in sorted(os.listdir(generations_base)):
            file_ = os.path.join(generations_base, generation_file)
            all_generations.append(np.load(file_))
        # Generations, losses are shape [num_prompts, num_trials, suffix_len].
        all_generations = np.stack(all_generations, axis=1)

        all_losses = []
        for losses_file in sorted(os.listdir(losses_base)):
            file_ = os.path.join(losses_base, losses_file)
            all_losses.append(np.load(file_))
        all_losses = np.stack(all_losses, axis=1)

    for generations_per_prompt in [1, 5, 10, 20, 50, all_generations.shape[1]]: #5, 10, 20, 50,
        limited_generations = all_generations[:, :generations_per_prompt, :]
        limited_losses = all_losses[:, :generations_per_prompt, :]

        print(limited_losses.shape)
        
        axis0 = np.arange(all_generations.shape[0])  #1000
        axis1 = limited_losses.argmin(1).reshape(-1)  #1000
        generations_one = limited_generations[axis0, axis1, :]
        batch_losses = limited_losses[axis0, axis1]
        order = np.argsort(batch_losses.flatten())
        generations_order = generations_one[order]
        write_guesses_order(generations_per_prompt, batch_losses, generations_one)


        precision, percision_multi, error_k, ham_dist = metric_print(generations_one, all_generations, generations_per_prompt, generations_order, order, _val_set_num.value)
        end_t = time.time()
        print('time cost:',end_t-start_t)
        # FOR TESTING !
        # def is_memorization(guesses, answers):
        #     return np.all(guesses==answers, axis=-1)
        #
        # answers = np.load(os.path.join(_DATASET_DIR.value, "val_dataset.npy"))[:, -50:].astype(np.int64)
        # print(guesses.shape, answers.shape)
        # print(np.sum(is_memorization(guesses, answers)) / 100)


if __name__ == "__main__":
    app.run(main)
