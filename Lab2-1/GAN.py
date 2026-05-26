import numpy as np
from numpy import hstack,zeros,ones
from numpy.random import rand ,randn
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot


# 定义生成器函数
def define_generator(latent_dim,n_outputs=2):
    model=Sequential()
    model.add(Dense(30,activation='relu',kernel_initializer='he_uniform',input_dim=latent_dim))
    model.add(Dense(n_outputs,activation='linear'))
    return model
    
# 定义判别器函数
def define_discriminator(n_inputs=2):
    model=Sequential()
    model.add(Dense(50,activation='relu',kernel_initializer='he_uniform',input_dim=n_inputs))
    model.add(Dense(1,activation='sigmoid'))    # sigmoid适合二分类问题
    # 编译模型
    model.compile(loss='binary_crossentropy',   # 损失函数（给模型看的，本损失函数-二元交叉熵，适合二分类问题）
                  optimizer='adam',             # 优化器（adam自动调整学习率，收敛速度快，适合大多数场景）
                  metrics=['accuracy'])         # 评估指标（给人类看的，人类最后关心的东西）
    return model

# 定义一个完整的生成对抗网络模型
def define_gan(generator,discriminator):
    # 锁定判别器权重，不让其参与训练
    discriminator.trainable=False
    model=Sequential()
    # 添加生成器
    model.add(generator)
    # 添加判别器
    model.add(discriminator)
    # 编译生成对抗网络
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model


# 编写函数用于生成真实样本
def generate_real_samples(n):
    x1=rand(n)*6-3
    x2=np.sin(x1)
    x1=x1.reshape(n,1)
    x2=x2.reshape(n,1)
    x=hstack((x1,x2))
    y=ones((n,1))
    return x,y

# 编写函数用于在隐空间中生成随机点
def generate_latent_points(latent_dim,n):
    x_input=randn(latent_dim*n)
    x_input=x_input.reshape(n,latent_dim)
    return x_input


# 编写函数令生成器使用隐空间点生成伪造样本
def generate_fake_samples(generator,latent_dim,n):
    x_input=generate_latent_points(latent_dim,n)
    x=generator.predict(x_input)
    y=zeros((n,1))
    return x,y
    

# 编写函数用于评估判别器的性能+绘制结果
def summarize_performance(epoch,generator,discriminator,latent_dim,n=100):
    x_real,y_real=generate_real_samples(n)
    _,acc_real=discriminator.evaluate(x_real,y_real,verbose=0) # verbose=0表示不打印评估过程
    x_fake,y_fake=generate_fake_samples(generator,latent_dim,n)
    _,acc_fake=discriminator.evaluate(x_fake,y_fake,verbose=0)
    print(epoch,acc_real,acc_fake)
    pyplot.scatter(x_real[:,0],x_real[:,1],color='red')
    pyplot.scatter(x_fake[:,0],x_fake[:,1],color='blue')
    filename='E:/self-cultivation/Python/人工智能安全/GAN/pictures/plot_%03d.png'%(epoch+1)
    pyplot.savefig(filename)
    pyplot.close()
    
# 编写训练函数
def train(g_model,d_model,gan_model,latent_dim,n_epochs=8000,n_batch=128,n_eval=2000):
    half_batch=int(n_batch/2)
    for i in range(n_epochs):
        if (i+1)%100==0:
            print('Epoch %d/%d'%(i+1,n_epochs))
        x_real,y_real=generate_real_samples(half_batch)
        x_fake,y_fake=generate_fake_samples(g_model,latent_dim,half_batch)
        d_model.train_on_batch(x_real,y_real)
        d_model.train_on_batch(x_fake,y_fake)
        x_gan=generate_latent_points(latent_dim,n_batch)
        y_gan=ones((n_batch,1))
        gan_model.train_on_batch(x_gan,y_gan)
        if (i+1)%n_eval==0:
            summarize_performance(i,g_model,d_model,latent_dim)
            
# 训练
latent_dim=6
discriminator=define_discriminator()
generator=define_generator(latent_dim)
gan_model=define_gan(generator,discriminator)
train(generator,discriminator,gan_model,latent_dim)

# 代码2-1 基于生成对抗网络的sin曲线样本模拟
# import numpy as np
# from numpy import hstack, zeros, ones
# from numpy.random import rand, randn
# from keras.models import Sequential
# from keras.layers import Dense
# from matplotlib import pyplot

# # 定义一个独立的生成器模型
# def define_generator(latent_dim, n_outputs=2):
#     model = Sequential()
#     model.add(Dense(30, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
#     model.add(Dense(n_outputs, activation='linear'))
#     return model

# # 定义一个独立的判别器模型
# def define_discriminator(n_inputs=2):
#     model = Sequential()
#     model.add(Dense(50, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
#     model.add(Dense(1, activation='sigmoid'))
#     # 编译模型
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# # 定义一个完整的生成对抗网络模型，即组合的生成器和判别器模型，以更新生成器
# def define_gan(generator, discriminator):
#     # 锁定判别器的权重，使它们不参与训练
#     discriminator.trainable = False
#     model = Sequential()
#     # 添加生成器
#     model.add(generator)
#     # 添加判别器
#     model.add(discriminator)
#     # 编译生成对抗网络模型
#     model.compile(loss='binary_crossentropy', optimizer='adam')
#     return model

# # 生成带有类标签的n个真实样本
# def generate_real_samples(n):
#     X1 = rand(n)*6 - 3
#     X2 = np.sin(X1)  # 生成正弦曲线
#     X1 = X1.reshape(n, 1)
#     X2 = X2.reshape(n, 1)
#     X = hstack((X1, X2))
#     y = ones((n, 1))
#     return X, y

# # 在隐空间中生成一些点作为生成器的输入
# def generate_latent_points(latent_dim, n):
#     x_input = randn(latent_dim * n)
#     x_input = x_input.reshape(n, latent_dim)
#     return x_input

# # 使用生成器生成带有类标签的n个伪造样本
# def generate_fake_samples(generator, latent_dim, n):
#     x_input = generate_latent_points(latent_dim, n)
#     X = generator.predict(x_input)
#     y = zeros((n, 1))
#     return X, y

# # 评估判别器并绘制真实样本和伪造样本
# def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
#     x_real, y_real = generate_real_samples(n)
#     _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
#     x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
#     _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
#     print(epoch, acc_real, acc_fake)
#     pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
#     pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
#     filename = 'E:/self-cultivation/Python/人工智能安全/GAN/pictures/plot_%03d.png'%(epoch+1)
#     pyplot.savefig(filename)
#     pyplot.close()

# # 训练生成对抗网络
# def train(g_model, d_model, gan_model, latent_dim, n_epochs=8000, n_batch=128, n_eval=2000):
#     half_batch = int(n_batch / 2)
#     for i in range(n_epochs):
#         x_real, y_real = generate_real_samples(half_batch)
#         x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#         # 更 新 判 别 器
#         # discriminator.trainable = True  # 仅较新 Keras 版本必须添加
#         d_model.train_on_batch(x_real, y_real)
#         d_model.train_on_batch(x_fake, y_fake)
#         # discriminator.trainable = False  # 仅较新 Keras 版本必须添加
#         x_gan = generate_latent_points(latent_dim, n_batch)
#         y_gan = ones((n_batch, 1))
#         gan_model.train_on_batch(x_gan, y_gan)
#         if (i + 1) % n_eval == 0:
#             summarize_performance(i, g_model, d_model, latent_dim)

# # 隐空间的维度
# latent_dim = 6
# discriminator = define_discriminator()
# generator = define_generator(latent_dim)
# gan_model = define_gan(generator, discriminator)
# train(generator, discriminator, gan_model, latent_dim)

