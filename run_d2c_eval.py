import os
import shutil
import sys
import subprocess
from pathlib import Path

def main():
    args = [a for a in sys.argv[1:] if a != "--use-input-screenshot"]
    use_input_screenshot = len(args) != len(sys.argv) - 1

    if len(args) < 1:
        print("Usage: python run_d2c_eval.py <run_dir> [--use-input-screenshot]")
        sys.exit(1)

    run_dir = Path(args[0]).resolve()
    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist.")
        sys.exit(1)

    d2c_dir = Path("/work/tesi_dbilardello/Design2Code/Design2Code")
    if not d2c_dir.exists():
        print(f"Error: {d2c_dir} does not exist.")
        sys.exit(1)

    run_name = run_dir.name
    print(f"Preparing Design2Code evaluation for {run_name}...")

    refs_dir = d2c_dir / f"testset_final_{run_name}"
    preds_dir = d2c_dir / "predictions_final" / run_name

    if refs_dir.exists():
        shutil.rmtree(refs_dir)
    if preds_dir.exists():
        shutil.rmtree(preds_dir)

    refs_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for sample_dir in run_dir.iterdir():
        if not sample_dir.is_dir(): continue
        sample_name = sample_dir.name
        
        ref_html = sample_dir / "reference.html"
        pred_html = sample_dir / "generated.html"
        input_png = sample_dir / "input.png"

        if ref_html.exists():
            shutil.copy(ref_html, refs_dir / f"{sample_name}.html")
            if use_input_screenshot and input_png.exists():
                # Ground truth = the actual screenshot given to the model for
                # generation, instead of a fresh D2C re-render of reference.html
                # (which can visually differ: fonts, blocked external resources,
                # renderer differences).
                shutil.copy(input_png, refs_dir / f"{sample_name}.png")
            if pred_html.exists():
                shutil.copy(pred_html, preds_dir / f"{sample_name}.html")
            else:
                # Create an empty HTML file so it counts as a penalty (0 score) instead of being skipped
                with open(preds_dir / f"{sample_name}.html", "w") as f:
                    f.write("<html><body></body></html>")
            copied += 1

    print(f"Copied {copied} HTML pairs.")

    # Try to copy rick.jpg (used by some datasets as fallback)
    rick_path_1 = d2c_dir.parent / "testset_final" / "rick.jpg"
    rick_path_2 = d2c_dir / "testset_final" / "rick.jpg"
    if rick_path_1.exists():
        shutil.copy(rick_path_1, refs_dir / "rick.jpg")
    elif rick_path_2.exists():
        shutil.copy(rick_path_2, refs_dir / "rick.jpg")

    eval_script_name = f"run_eval_{run_name}.py"
    eval_script = d2c_dir / eval_script_name
    script_content = '''from metrics.visual_score import visual_eval_v3_multi
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import json
import os
import sys
import traceback

def safe_visual_eval(input_list, debug=False):
    """One pathological prediction must not abort the whole evaluation.

    Truncated generations often leave hundreds of unclosed tags, which
    BeautifulSoup nests cumulatively; extract_text_recursive then blows the
    recursion limit. joblib propagates that out of the worker and kills the
    run, so a single bad file used to waste hours of completed scoring.
    Raise the limit for legitimately deep pages, and score anything still
    unprocessable as 0 - the same penalty this script already applies to a
    missing prediction.
    """
    sys.setrecursionlimit(10000)
    try:
        return visual_eval_v3_multi(input_list, debug=debug)
    except Exception:
        print("[skipped] %s\\n%s" % (input_list[1], traceback.format_exc()), flush=True)
        return None

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def print_multi_score(multi_score):
    _, final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("Block-Match: ", final_size_score)
    print("Text: ", final_matched_text_score)
    print("Position: ", final_position_score)
    print("Color: ", final_text_color_score)
    print("CLIP: ", final_clip_score)
    print("--------------------------------\\n")

if __name__ == "__main__":
    debug = False
    multiprocessing = True
    reference_dir = "testset_final_screencoder"
    test_dirs = {"screencoder": "predictions_final/screencoder"}

    file_name_list = []
    for filename in os.listdir(reference_dir):
        if filename.endswith(".html"):
            if all([os.path.exists(os.path.join(test_dirs[key], filename)) for key in test_dirs]):
                file_name_list.append(filename)

    print ("total #egs: ", len(file_name_list))

    input_lists = []
    for filename in file_name_list:
        input_pred_list = [os.path.join(test_dirs[key], filename) for key in test_dirs]
        original = os.path.join(reference_dir, filename)
        input_lists.append([input_pred_list, original])

    if multiprocessing:
        with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
            n_jobs = int(os.environ.get("D2C_N_JOBS", "8"))
            return_score_lists = list(tqdm(Parallel(n_jobs=n_jobs)(delayed(safe_visual_eval)(input_list, debug=debug) for input_list in input_lists), total=len(input_lists)))
    else:
        return_score_lists = []
        for input_list in tqdm(input_lists):
            return_score_lists.append(visual_eval_v3_multi(input_list, debug=debug))
    
    res_dict = {key: {} for key in test_dirs}

    for i, filename in enumerate(file_name_list):
        idx = 0
        return_score_list = return_score_lists[i]
        if return_score_list:
            for key in test_dirs:
                if multiprocessing:
                    matched, final_score, multi_score = return_score_list[idx]
                else:
                    matched = return_score_list[idx][0]
                    final_score = return_score_list[idx][1]
                    multi_score = return_score_list[idx][2]
                idx += 1
                current_score = [final_score] + [item for item in multi_score]
                res_dict[key][filename] = current_score
        else:
            for key in test_dirs:
                res_dict[key][filename] = [0, 0, 0, 0, 0, 0]

    with open("metrics/res_dict_screencoder.json", "w") as f:
        json.dump(res_dict, f, indent=4)

    for key in test_dirs:
        print(key)
        values = list(res_dict[key].values())
        current_res = np.mean(np.array(values), axis=0)
        print_multi_score(current_res)
'''.replace('screencoder', run_name)
    with open(eval_script, "w") as f:
        f.write(script_content)

    print(f"Running D2C official evaluation script... (use_input_screenshot={use_input_screenshot})")
    env = os.environ.copy()
    parent_dir = os.path.dirname(d2c_dir)
    env["PYTHONPATH"] = parent_dir + os.pathsep + env.get("PYTHONPATH", "")
    if use_input_screenshot:
        env["D2C_SKIP_GT_RERENDER"] = "1"
    subprocess.run([sys.executable, eval_script_name], cwd=d2c_dir, env=env, check=True)

    print(f"Evaluation complete! Results are saved in {d2c_dir}/metrics/res_dict_{run_name}.json")

if __name__ == "__main__":
    main()
