import torch
from torch import optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from helper import *
from model.generator import SkipEncoderDecoder, input_noise
import argparse


def remove_watermark(
    image_path, mask_path, max_dim, reg_noise, input_depth,
    lr, show_step, training_steps, tqdm_length=100
):
    DTYPE = torch.FloatTensor
    has_set_device = False

    if torch.cuda.is_available():
        device = 'cuda'
        has_set_device = True
        print("Setting Device to CUDA...")

    try:
        if torch.backends.mps.is_available():
            device = 'mps'
            has_set_device = True
            print("Setting Device to MPS...")
    except Exception as e:
        print(f"Your version of pytorch might be too old, which does not support MPS. Error: \n{e}")
        pass

    if not has_set_device:
        device = 'cpu'
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5
    ).type(DTYPE).to(device)

    objective = torch.nn.MSELoss().type(DTYPE).to(device)
    optimizer = optim.Adam(generator.parameters(), lr)

    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)
    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    losses = []

    print('\nStarting training...\n')
    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_saved

        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)
        loss = objective(output * mask_var, image_var * mask_var)
        losses.append(loss.item())

        loss.backward()

        if step % show_step == 0:
            output_image = torch_to_np_array(output)
            visualize_sample(image_np, output_image, nrow=2, size_factor=10)

            step_output_path = f'{image_path.split("/")[-1].split(".")[-2]}-step-{step}.jpg'
            pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))
            pil_image.save(step_output_path)
            print(f'Saved step {step} output image to: "{step_output_path}"')

        progress_bar.set_postfix(Loss=loss.item())
        optimizer.step()

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

    output_image = torch_to_np_array(output)
    visualize_sample(output_image, nrow=1, size_factor=10)

    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))
    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')
    pil_image.save(output_path)
        
        
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    # 添加参数：输出图像的最大尺寸
    parser.add_argument('--max-dim', type=int,
                        default=512, help='Max dimension of the final output image')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用remove_watermark函数，传入命令行参数
    remove_watermark(
        image_path=args.image_path,
        mask_path=args.mask_path,
        max_dim=args.max_dim,
        show_step=args.show_step,
        reg_noise=args.reg_noise,
        input_depth=args.input_depth,
        lr=args.lr,
        training_steps=args.training_steps,
    )