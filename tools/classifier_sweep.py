import os
import glob
import pandas as pd
from ultralytics import YOLO

def main():
    model_path = 'models/locked/v2_baseline_001/seatbelt_classifier/best.pt'
    dataset_dir = 'datasets/derived/v2_pretrain_pending_approval/seatbelt_classifier/val'
    
    print(f"Loading {model_path}...")
    model = YOLO(model_path)
    
    classes = ['seatbelt_fastened', 'seatbelt_unfastened', 'uncertain_or_occluded']
    results_list = []
    
    print("Running inference to collect raw probabilities...")
    for true_cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
            
        img_paths = glob.glob(os.path.join(cls_dir, '*.jpg')) + glob.glob(os.path.join(cls_dir, '*.png'))
        
        # Batch predict for speed
        batch_size = 32
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            results = model.predict(batch_paths, verbose=False)
            
            for path, r in zip(batch_paths, results):
                probs = r.probs.data.cpu().numpy() # array of probabilities
                results_list.append({
                    'image': os.path.basename(path),
                    'true_class': cls_name,
                    'prob_fastened': probs[0],
                    'prob_unfastened': probs[1],
                    'prob_uncertain': probs[2],
                    'raw_pred': classes[r.probs.top1]
                })
                
    df = pd.DataFrame(results_list)
    df.to_csv('reports/calibration/seatbelt_classifier_raw_predictions.csv', index=False)
    print(f"Collected {len(df)} predictions.")
    
    # Now perform the sweep
    fastened_thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    unfastened_thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    sweep_results = []
    
    for ft in fastened_thresholds:
        for ut in unfastened_thresholds:
            # apply reject policy
            def apply_policy(row):
                if row['prob_fastened'] > row['prob_unfastened'] and row['prob_fastened'] > row['prob_uncertain']:
                    # raw pred is fastened
                    if row['prob_fastened'] >= ft: return 'seatbelt_fastened'
                    else: return 'uncertain_or_occluded'
                elif row['prob_unfastened'] > row['prob_fastened'] and row['prob_unfastened'] > row['prob_uncertain']:
                    # raw pred is unfastened
                    if row['prob_unfastened'] >= ut: return 'seatbelt_unfastened'
                    else: return 'uncertain_or_occluded'
                else:
                    return 'uncertain_or_occluded'
                    
            preds = df.apply(apply_policy, axis=1)
            
            # calculate metrics
            total = len(df)
            rejected = (preds == 'uncertain_or_occluded').sum()
            reject_rate = rejected / total
            coverage = 1.0 - reject_rate
            
            # false unfastened = True is Fastened, Pred is Unfastened
            false_unfastened = ((df['true_class'] == 'seatbelt_fastened') & (preds == 'seatbelt_unfastened')).sum()
            
            # false fastened = True is Unfastened, Pred is Fastened
            false_fastened = ((df['true_class'] == 'seatbelt_unfastened') & (preds == 'seatbelt_fastened')).sum()
            
            # accuracy (overall)
            correct = (df['true_class'] == preds).sum()
            accuracy = correct / total
            
            # P, R for Unfastened
            tp_unfast = ((df['true_class'] == 'seatbelt_unfastened') & (preds == 'seatbelt_unfastened')).sum()
            fp_unfast = ((df['true_class'] != 'seatbelt_unfastened') & (preds == 'seatbelt_unfastened')).sum()
            fn_unfast = ((df['true_class'] == 'seatbelt_unfastened') & (preds != 'seatbelt_unfastened')).sum()
            p_unfast = tp_unfast / (tp_unfast + fp_unfast) if (tp_unfast + fp_unfast) > 0 else 0
            r_unfast = tp_unfast / (tp_unfast + fn_unfast) if (tp_unfast + fn_unfast) > 0 else 0
            f1_unfast = 2 * p_unfast * r_unfast / (p_unfast + r_unfast) if (p_unfast + r_unfast) > 0 else 0
            
            # P, R for Fastened
            tp_fast = ((df['true_class'] == 'seatbelt_fastened') & (preds == 'seatbelt_fastened')).sum()
            fp_fast = ((df['true_class'] != 'seatbelt_fastened') & (preds == 'seatbelt_fastened')).sum()
            fn_fast = ((df['true_class'] == 'seatbelt_fastened') & (preds != 'seatbelt_fastened')).sum()
            p_fast = tp_fast / (tp_fast + fp_fast) if (tp_fast + fp_fast) > 0 else 0
            r_fast = tp_fast / (tp_fast + fn_fast) if (tp_fast + fn_fast) > 0 else 0
            f1_fast = 2 * p_fast * r_fast / (p_fast + r_fast) if (p_fast + r_fast) > 0 else 0
            
            macro_f1 = (f1_unfast + f1_fast) / 2  # simplified macro F1 for main classes
            
            sweep_results.append({
                'fastened_threshold': ft,
                'unfastened_threshold': ut,
                'coverage': coverage,
                'accuracy': accuracy,
                'macro_f1_main_classes': macro_f1,
                'false_unfastened_count': false_unfastened,
                'false_fastened_count': false_fastened,
                'unknown_reject_rate': reject_rate
            })
            
    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv('reports/calibration/seatbelt_classifier_threshold_sweep.csv', index=False)
    print("Saved reports/calibration/seatbelt_classifier_threshold_sweep.csv")
    
if __name__ == "__main__":
    main()
