import eval_util
import tensorflow as tf
import os
import utils

class EvalMetricsHook(tf.train.SessionRunHook):
    def __init__(self,model_dir):
        super(EvalMetricsHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):

        self.predictions = tf.get_collection("predictions")[0]
        self.labels=  tf.get_collection("labels")[0]
        self.global_step = tf.train.get_global_step()
    def before_run(self,run_context):
        self.writer = tf.summary.FileWriter(os.path.join(self.model_dir,'eval'),
                                      run_context.session.graph)

        return tf.train.SessionRunArgs({'predictions':self.predictions,
                                        'labels':self.labels,
                                        'step':self.global_step})
    def after_run(self,run_context, run_values):

        predictions_val = run_values.results['predictions']
        labels_val = run_values.results['labels']
        step = run_values.results['step']
        hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
        perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,labels_val)
        gap = eval_util.calculate_gap(predictions_val, labels_val)

        self.writer.add_summary(
            utils.MakeSummary("model/Training_Hit@1", hit_at_one),
            step)
        self.writer.add_summary(
            utils.MakeSummary("model/Training_Perr", perr),step)
        self.writer.add_summary(
            utils.MakeSummary("model/Training_GAP", gap), step)

    def end(self,session):

        pass
