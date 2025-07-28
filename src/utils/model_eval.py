import numpy as np


def get_num_classes_by_dataset_name(dataset_name):
    if "modelnet40" in dataset_name.lower():
        num_classes = 40
    elif "shapenet" in dataset_name.lower():
        num_classes = 16
    else:
        raise NotImplementedError("Dataset not supported")

    return num_classes


class PerfTrackVal:
    """
    Records epoch wise performance for validation
    """

    def __init__(self):
        self.all = []
        self.class_seen = None
        self.class_corr = None

    def update(self, logits, ground_truth_label):
        correct = self.get_correct_list(logits, ground_truth_label)
        self.all.extend(correct)
        self.update_class_see_corr(logits, ground_truth_label)

    def agg(self):
        perf = {
            "acc": self.get_avg_list(self.all),
            "class_acc": float(
                np.mean(
                    np.array(self.class_corr)
                    / np.array(self.class_seen, dtype=np.float32)
                )
            ),
        }
        return perf

    def update_class_see_corr(self, logit, label):
        if self.class_seen is None:
            num_class = logit.shape[1]
            self.class_seen = [0] * num_class
            self.class_corr = [0] * num_class

        pred_label = logit.argmax(axis=1).to("cpu").tolist()
        for _pred_label, _label in zip(pred_label, label):
            self.class_seen[_label] += 1
            if _pred_label == _label:
                self.class_corr[_pred_label] += 1

    @staticmethod
    def get_correct_list(logit, label):
        label = label.to(logit.device)
        pred_class = logit.argmax(axis=1)
        return (label == pred_class).to("cpu").tolist()

    @staticmethod
    def get_avg_list(all_list):
        for x in all_list:
            assert isinstance(x, bool)
        return sum(all_list) / len(all_list)
