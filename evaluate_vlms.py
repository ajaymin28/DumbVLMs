from tools.evaluation.Evaluator import Evaluator
from benchmark.TaskLoader import TaskLoader

if __name__=="__main__":

    for modelname in ["llava_ov", "internvl2_5", "qwen2_5_vl"]:
        
        eval = Evaluator(model="llava_ov")

        task_loader = TaskLoader(task_type="shape_matching")
        eval.evaluate_shapematching(task_loader=task_loader)

        task_loader = TaskLoader(task_type="odd_one_out")
        eval.evaluate_oddoneout(task_loader=task_loader)

        task_loader = TaskLoader(task_type="rotation_reasoning")
        eval.evaluate_rotationreasoning(task_loader=task_loader)