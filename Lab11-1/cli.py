import click
import sys
sys.path.insert(1, 'Lab11-1')
import af_attack

@click.group()
def cli():
    pass

@cli.group()
def attribute_inference():
    pass

@attribute_inference.command(help='Load trained target and attack model')
def pretrained_dummy():
    click.echo('Performing Attribute Inference with trained target and attack model')
    af_attack.perform_pretrained_dummy()

@attribute_inference.command(help='Train target and attack model')
@click.option('-t', '--target_epochs', default=30, help='Number of training epochs for the target model')
@click.option('-a', '--attack_epochs', default=50, help='Number of training epochs for the attack model')
def train_dummy(target_epochs, attack_epochs):
    click.echo('Performing Attribute Inference with training of target and attack model')
    af_attack.perform_train_dummy(target_epochs, attack_epochs)

@attribute_inference.command(help='Supply own target model and train attack model')
@click.option('-c', '--class_file', required=True, type=str, help='File that holds the target models nn.Module class')
@click.option('-s', '--state_path', required=True, type=str, help='Path of the state dictionary')
@click.option('-d', '--dimension', required=True, type=int, help='Flattend dimension of the layer used as attack model input')
@click.option('-a', '--attack_epochs', default=50, type=int, help='Number of training epochs for the attack model')
def supply_target(class_file, state_path, dimension, attack_epochs):
    click.echo('Performing Attribute Inference')
    af_attack.perform_supply_target(class_file, state_path, dimension, attack_epochs)

if __name__ == "__main__":
    cli()