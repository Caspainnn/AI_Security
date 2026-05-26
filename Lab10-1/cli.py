import click
import sys
sys.path.insert(1, 'Membership-Inference')
import attack


# 创建命令组
@click.group()
def cli():
    pass


@cli.group(help='Membership Inference Attack commands')
def membership_inference():
    """Membership Inference Attack commands"""
    pass


# 执行预训练模型的成员推理攻击
@membership_inference.command(help='Perform Membership Inference with pretrained model')
@click.option('--dataset', default='CIFAR10', type=str, help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data_path', default='./Lab10-1/Membership-Inference/data', type=str, help='Path to store data')
@click.option('--model_path', default='./Lab10-1/Membership-Inference/best_models', type=str, help='Path to load model checkpoints')
def pretrained_dummy(dataset,data_path,model_path):
    click.echo('Preforming Membership Inference')
    attack.create_attack(dataset,data_path,model_path,False,False,False,False,False,False)
    

# 执行成员推理攻击并训练模型
@membership_inference.command(help='Perform Membership Inference with model training')
@click.option('--dataset', default='CIFAR10', type=str, help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data_path', default='./Lab10-1/Membership-Inference/data', type=str, help='Path to store data')
@click.option('--model_path', default='./Lab10-1/Membership-Inference/best_models', type=str, help='Path to save model checkpoints')
def train_dummy(dataset,data_path,model_path):
    click.echo('Preforming Mmbership Inference')
    attack.create_attack(dataset,data_path,model_path,True,True,False,False,False,False)
    
# 成员推理攻击+训练模型 可选择数据增强，使用 top 3 后验概率 初始化参数 详细输出
@membership_inference.command(help='Membership Inference Attack with training enabled+augmentation, topk posteriors, parameter initialization and verbose')
@click.option('--dataset', default='CIFAR10', type=str, help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data_path', default='./Lab10-1/Membership-Inference/data', type=str, help='Path to store data')
@click.option('--model_path', default='./Lab10-1/Membership-Inference/best_models', type=str, help='Path to save or load model checkpoints')
@click.option('--need_augm', is_flag=True, help='Use data augmentation on target and shadow training set')
# is_flag=True，那么这个选项在命令行中 不需要再跟参数值。默认是 False，只有当你写上它时才变成 True
@click.option('--need_topk', is_flag=True, help='Enable using Top 3 posteriors for attack data')
@click.option('--param_init', is_flag=True, help='Enable custom model params initialization')
@click.option('--verbose', is_flag=True, help='Add verbosity')
def train_plus_dummy(dataset,data_path,model_path,need_augm,need_topk,param_init,verbose):
    click.echo('Preforming Mmbership Inference')
    attack.create_attack(dataset,data_path,model_path,True,True,need_augm,need_topk,param_init,verbose)


if __name__ == '__main__':
    cli()