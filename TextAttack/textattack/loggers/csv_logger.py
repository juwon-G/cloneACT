"""
Attack Logs to CSV
========================
"""

import csv

import pandas as pd

from textattack.shared import AttackedText, logger

from .logger import Logger
import pickle
import os


class CSVLogger(Logger):
    """Logs attack results to a CSV."""

    def __init__(self, filename="results.csv", color_method="file"):
        logger.info(f"Logging to CSV at path {filename}")
        self.filename = filename
        self.color_method = color_method
        self.row_list = []
        self._flushed = True
        self.count = 0

    def log_attack_result(self, result):
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output[0],
            "perturbed_output": result.perturbed_result.output[0],
            "org_decode": result.original_result.raw_output[2],
            "org_tr_decode": result.original_result.raw_output[3],
            "adv_decode": result.perturbed_result.raw_output[2],
            "adv_tr_decode": result.perturbed_result.raw_output[3],
            "org_att_enc": result.original_result.raw_output[4],
            "org_att_dec": result.original_result.raw_output[5],
            "org_att_cro": result.original_result.raw_output[6],
            "adv_att_enc": result.perturbed_result.raw_output[4],
            "adv_att_dec": result.perturbed_result.raw_output[5],
            "adv_att_cro": result.perturbed_result.raw_output[6],
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        base, ext = os.path.splitext(self.filename)
        filename = f"{base}_{self.count}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(row, f)
        self.count += 1
        #self.row_list.append(row)
        self._flushed = False

    def flush(self):
        #self.df = pd.DataFrame.from_records(self.row_list)
        #self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def close(self):
        # self.fout.close()
        super().close()

    def __del__(self):
        if not self._flushed:
            logger.warning("CSVLogger exiting without calling flush().")
