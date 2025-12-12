#!/usr/bin/env python3
"""
Analyze Validation and Test Sets
Evaluate if val/test sets need rebalancing or augmentation
"""

from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValTestAnalyzer:
    """Analyze validation and test sets for ML best practices"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        self.test_path = self.dataset_path / "test"
    
    def count_all_images(self):
        """Count images in all splits"""
        counts = {}
        
        for split_name, split_path in [("train", self.train_path), ("val", self.val_path), ("test", self.test_path)]:
            counts[split_name] = {}
            total_split = 0
            
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        class_count = len([f for f in class_dir.glob('*.jpeg')])
                        counts[split_name][class_dir.name] = class_count
                        total_split += class_count
            
            counts[split_name]['total'] = total_split
        
        return counts
    
    def analyze_splits(self, counts):
        """Analyze split quality and provide recommendations"""
        
        # Calculate totals
        train_total = counts['train']['total']
        val_total = counts['val']['total']
        test_total = counts['test']['total']
        dataset_total = train_total + val_total + test_total
        
        # Calculate percentages
        train_pct = (train_total / dataset_total) * 100
        val_pct = (val_total / dataset_total) * 100
        test_pct = (test_total / dataset_total) * 100
        
        # Analyze class balance within each split
        analysis = {
            'dataset_totals': {
                'train': train_total,
                'val': val_total,
                'test': test_total,
                'total': dataset_total
            },
            'split_percentages': {
                'train': train_pct,
                'val': val_pct,
                'test': test_pct
            },
            'class_balance_analysis': {},
            'recommendations': [],
            'issues_found': []
        }
        
        # Analyze each split
        for split_name in ['train', 'val', 'test']:
            split_data = counts[split_name]
            
            # Get class counts (excluding 'total' and augmented folders)
            normal_count = split_data.get('NORMAL', 0)
            pneumonia_count = split_data.get('PNEUMONIA', 0)
            
            if split_name == 'train':
                # For training, include augmented NORMAL
                normal_aug = split_data.get('NORMAL_augmented', 0)
                normal_total = normal_count + normal_aug
            else:
                normal_total = normal_count
            
            # Calculate balance ratio
            if normal_total > 0 and pneumonia_count > 0:
                ratio = pneumonia_count / normal_total
                balance_status = "balanced" if 0.5 <= ratio <= 2.0 else "imbalanced"
            else:
                ratio = float('inf') if normal_total == 0 else 0
                balance_status = "severely_imbalanced"
            
            analysis['class_balance_analysis'][split_name] = {
                'normal_count': normal_total,
                'pneumonia_count': pneumonia_count,
                'ratio': f"1:{ratio:.2f}",
                'balance_status': balance_status
            }
        
        # Check for issues and generate recommendations
        
        # 1. Validation set size
        if val_total < 100:
            analysis['issues_found'].append("Validation set too small")
            analysis['recommendations'].append(f"Increase validation set size (current: {val_total}, recommended: 100-500)")
        
        # 2. Validation set balance
        val_balance = analysis['class_balance_analysis']['val']
        if val_balance['balance_status'] != 'balanced':
            analysis['issues_found'].append("Validation set imbalanced")
            analysis['recommendations'].append(f"Balance validation set (current ratio: {val_balance['ratio']})")
        
        # 3. Test set balance
        test_balance = analysis['class_balance_analysis']['test']
        if test_balance['balance_status'] != 'balanced':
            analysis['issues_found'].append("Test set imbalanced")
            analysis['recommendations'].append(f"Balance test set (current ratio: {test_balance['ratio']})")
        
        # 4. Split proportions
        if val_pct < 10:
            analysis['issues_found'].append("Validation set proportion too small")
            analysis['recommendations'].append(f"Increase validation set to 10-15% of dataset (current: {val_pct:.1f}%)")
        
        if test_pct < 15:
            analysis['issues_found'].append("Test set proportion small")
            analysis['recommendations'].append(f"Consider increasing test set to 15-20% of dataset (current: {test_pct:.1f}%)")
        
        return analysis
    
    def generate_rebalancing_plan(self, analysis):
        """Generate a plan to rebalance val/test sets"""
        
        plan = {
            'rebalancing_needed': len(analysis['issues_found']) > 0,
            'actions': [],
            'target_sizes': {},
            'redistribution_plan': {}
        }
        
        if not plan['rebalancing_needed']:
            return plan
        
        dataset_total = analysis['dataset_totals']['total']
        
        # Recommended target sizes
        target_val_size = max(100, int(dataset_total * 0.12))  # 12% for validation
        target_test_size = max(200, int(dataset_total * 0.18))  # 18% for test
        target_train_size = dataset_total - target_val_size - target_test_size
        
        plan['target_sizes'] = {
            'train': target_train_size,
            'val': target_val_size,
            'test': target_test_size
        }
        
        # For medical datasets, we want balanced val/test sets
        target_val_per_class = target_val_size // 2
        target_test_per_class = target_test_size // 2
        
        plan['redistribution_plan'] = {
            'val': {
                'NORMAL': target_val_per_class,
                'PNEUMONIA': target_val_per_class
            },
            'test': {
                'NORMAL': target_test_per_class,
                'PNEUMONIA': target_test_per_class
            }
        }
        
        # Generate specific actions
        current_val = analysis['dataset_totals']['val']
        current_test = analysis['dataset_totals']['test']
        
        if current_val < target_val_size:
            plan['actions'].append(f"Move {target_val_size - current_val} images from train to validation")
        
        if current_test < target_test_size:
            plan['actions'].append(f"Move {target_test_size - current_test} images from train to test")
        
        plan['actions'].append("Balance classes within validation set")
        plan['actions'].append("Balance classes within test set")
        plan['actions'].append("Ensure no data leakage between splits")
        
        return plan
    
    def run_analysis(self):
        """Run complete analysis"""
        logger.info("🔍 Analyzing validation and test sets...")
        
        # Count images
        counts = self.count_all_images()
        
        # Analyze splits
        analysis = self.analyze_splits(counts)
        
        # Generate rebalancing plan
        rebalancing_plan = self.generate_rebalancing_plan(analysis)
        
        # Combine results
        full_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'image_counts': counts,
            'analysis': analysis,
            'rebalancing_plan': rebalancing_plan
        }
        
        return full_report
    
    def print_analysis(self, report):
        """Print human-readable analysis"""
        
        print("🔍 VALIDATION & TEST SET ANALYSIS")
        print("=" * 50)
        
        # Current state
        counts = report['image_counts']
        analysis = report['analysis']
        
        print("📊 CURRENT DATASET DISTRIBUTION:")
        for split in ['train', 'val', 'test']:
            split_data = counts[split]
            total = split_data['total']
            pct = analysis['split_percentages'][split]
            print(f"   {split.upper()}: {total:,} images ({pct:.1f}%)")
            
            # Show class breakdown
            for class_name, count in split_data.items():
                if class_name != 'total':
                    print(f"     {class_name}: {count:,}")
        
        print("\\n⚖️  CLASS BALANCE ANALYSIS:")
        for split, balance_info in analysis['class_balance_analysis'].items():
            status_emoji = "✅" if balance_info['balance_status'] == 'balanced' else "⚠️"
            print(f"   {split.upper()}: {status_emoji} {balance_info['ratio']} ({balance_info['balance_status']})")
        
        # Issues and recommendations
        if analysis['issues_found']:
            print("\\n🚨 ISSUES FOUND:")
            for issue in analysis['issues_found']:
                print(f"   ❌ {issue}")
            
            print("\\n💡 RECOMMENDATIONS:")
            for rec in analysis['recommendations']:
                print(f"   ✅ {rec}")
        else:
            print("\\n✅ NO ISSUES FOUND - Val/Test sets are properly configured!")
        
        # Rebalancing plan
        plan = report['rebalancing_plan']
        if plan['rebalancing_needed']:
            print("\\n🔄 REBALANCING PLAN:")
            print("   Target sizes:")
            for split, size in plan['target_sizes'].items():
                print(f"     {split.upper()}: {size:,} images")
            
            print("\\n   Actions needed:")
            for action in plan['actions']:
                print(f"     • {action}")
        else:
            print("\\n✅ NO REBALANCING NEEDED!")


def main():
    """Main function"""
    dataset_path = "data/chest_xray_final"
    
    analyzer = ValTestAnalyzer(dataset_path)
    report = analyzer.run_analysis()
    
    # Print analysis
    analyzer.print_analysis(report)
    
    # Save detailed report
    with open('val_test_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\\n📄 Detailed report saved as 'val_test_analysis_report.json'")
    
    return report

if __name__ == "__main__":
    report = main()