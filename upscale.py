import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D,AveragePooling2D,Conv2DTranspose, Input, Concatenate, Add, BatchNormalization, Activation, MultiHeadAttention
import tensorflow_hub as hub
import tensorflow_text as text

class GaussianDiffusion:
    """Утилита для гауссовского диффузии.

    Args:
        beta_start: Начальное значение дисперсии
        beta_end: Конечное значение дисперсии
        timesteps: Количество временных шагов в процессе прямой, а затем обратной диффузии
    """

    def __init__(
        self, beta_start=1e-4, beta_end=0.02, timesteps=1000, clip_min=-1.0, clip_max=1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Определение линейного пространства дисперсии
        self.betas = betas = np.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=np.float64,  # Тут используется float64 для лучшей точности
        )
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)

        # Расчеты для диффузии q(x_t | x_{t-1}) и других
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)

        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)

        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1.0 - alphas_cumprod), dtype=tf.float32)

        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32)

        # Расчеты для апостериорной q(x_{t-1} | x_t, x_0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)

        # Обрезка расчета логарифма, так как апостериорная дисперсия равна 0 в начале цепочки диффузии
        self.posterior_log_variance_clipped = tf.constant(np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32)

        self.posterior_mean_coef1 = tf.constant(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),dtype=tf.float32,)

        self.posterior_mean_coef2 = tf.constant(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),dtype=tf.float32)

    def _extract(self, a, t, x_shape):
        """Извлекает некоторые коэффициенты в указанных временных шагах,
        затем изменяет форму на [batch_size, 1, 1, 1, 1, ...] совпадения форм.

        Args:
            a: Тензор для извлечения
            t: Временной шаг, для которого коэффициенты должны быть извлечены
            x_shape: Форма текущих выборок в батче
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])

    def q_mean_variance(self, x_start, t):
        """Извлекает среднее значение и дисперсию на текущем временном шаге.

        Args:
            x_start: Начальный образец (перед первым шагом диффузии)
            t: Текущий временной шаг
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        """Диффузия данных.

        Args:
            x_start: Начальный образец (перед первым шагом диффузии)
            t: Текущий временной шаг
            noise: Добавляемый гауссовский шум на текущем временном шаге
        Returns:
            Диффузионные образцы на временном шаге `t`
        """
        x_start_shape = tf.shape(x_start)
        
        return (
            self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
            * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Вычисляет среднее значение и дисперсию диффузии апостериорной q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Точка начала (образец) для вычисления апостериори
            x_t: Образец на временном шаге `t`
            t: Текущий временной шаг
        Returns:
            Апостериорное среднее значение и дисперсия на текущем временном шаге
        """

        x_t_shape = tf.shape(x_t)
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t_shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        """Выборка из модели диффузии.

        Args:
            pred_noise: Шум, предсказанный моделью диффузии
            x: Образцы на определенном временном шаге, для которого был предсказан шум
            t: Текущий временной шаг
            clip_denoised (bool): Нужно ли обрезать предсказанный шум в указанном диапазоне или нет.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)
        # Нет шума, когда t == 0
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1])
        
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


embedding_dims = 32
embedding_max_frequency = 1000.0

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, activation=keras.activations.swish)(x)
        x = layers.Conv2D(width/2, kernel_size=1, activation=keras.activations.swish)(x)
        x = layers.Conv2D(width/2, kernel_size=3, padding="same", activation=keras.activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=1, activation=keras.activations.swish)(x)
        x = layers.Add()([x, residual])
        return x
    return apply

def DownBlock(width, block_depth, time_embedded = True):
    width, att = width
    def apply(x):
        if time_embedded:
            x, skips, emb, annotation_embedding = x
        else:
            x, skips, annotation_embedding = x
        height = x.shape[1]
        if att:
            t_emb = layers.UpSampling2D(size=height, interpolation="nearest")(annotation_embedding)
            x = Concatenate()([x, t_emb])
            
        if time_embedded:    
            e = layers.UpSampling2D(size=height, interpolation="nearest")(emb)
            x = Concatenate()([x, e])
            
        for _ in range(1 if height == 128 else block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    width, att = width
    def apply(x):
        x, skips, annotation_embedding = x
        height = x.shape[1]
        if att:
            t_emb = layers.UpSampling2D(size=height, interpolation="nearest")(annotation_embedding)
            x = Concatenate()([x, t_emb])
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(1 if height == 128//2 else block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances_r = keras.Input(shape=(1,1,1))
    annotation_embedding = keras.Input(shape=(1, 1, 512))
    small_image = keras.Input(shape=(image_size, image_size, 3))
    
    x = noisy_images
    upsampled = small_image
    x = Concatenate()([x, upsampled])
    emb = layers.Lambda(sinusoidal_embedding)(noise_variances_r)

    skips = []
    
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips, emb, annotation_embedding])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1][0])(x)
        
    t_emb = layers.UpSampling2D(size=8, interpolation="nearest")(annotation_embedding)
    x = Concatenate()([x, t_emb])

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips, annotation_embedding])
        
    x = Concatenate()([x, noisy_images, upsampled])
    x = layers.Conv2D(64, kernel_size=1, padding = 'same',  activation=keras.activations.swish)(x)
    x = layers.Conv2D(3, kernel_size=1, padding = 'same', kernel_initializer="zeros" )(x)

    return keras.Model([noisy_images, noise_variances_r, annotation_embedding, small_image], x, name="residual_unet")


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        
        #используем технологию ema network - бегущее среднее для всех весов нейросети.
        #никак не влияет на обучаемую модель, но в процессе обучения обновляет веса ema модели.
        #в результате используем именно ema модель. Это стабилизирует результат, чинит любые шумы, "перекосы цветов" и т.д.
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.lr = 1e-4
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
    @tf.function
    def train_step(self, images, annotations):
        # 1. Найдем размер батча
        batch_size = tf.shape(images)[0]

        # 2. Получим случайные значечения временных шагов
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            
            small_images = tf.image.resize(images, (image_size//2, image_size//2))
            small_images = tf.image.resize(small_images, (image_size, image_size))
            
            # 3. Получим случайный шум. Его и будем добавлять к картинке
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)

            # 4. Применяем шум по правилам гауссовской диффузии
            images_t = self.gdf_util.q_sample(images, t, noise)

            # 5. Пропустим зашумленное изображение через нейросеть
            pred_noise = self.network([images_t, tf.reshape(t, shape = (-1, 1,1,1)), tf.reshape(annotations, shape = (-1, 1,1,512)), small_images], training=True)

            # 6. Расчитаем ошибку
            loss = tf.reduce_mean((noise - pred_noise)**2, axis = (1,2,3))

        # 7. Расчитаем градиенты
        gradients = tape.gradient(loss, self.network.trainable_weights)

        # 8. Обновим веса нейросети
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # 9. Обновим веса EMA сети
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return loss

    def plot_images(self, num_rows, num_cols, figsize, images):
        images = (tf.clip_by_value(images * 127 + 127, 0.0, 255.0).numpy().astype(np.uint8))
        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(images):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()
        
    def run_generation(self, num_rows=2, num_cols=8, figsize=(20, 10), annotation = " ", ema_mode = True, ex_rate = 0, from_annotation = False):
        num_images = num_rows * num_cols
        for image_batch, ann_batch in dataset.take(1):
            small_images = tf.image.resize(image_batch, (image_size//2, image_size//2))
            small_images = tf.image.resize(small_images, (image_size, image_size))[:num_images]
        self.plot_images(num_rows, num_cols, figsize, small_images*2.0)  
        
        if from_annotation:
            annotation = tf.expand_dims(annotation, axis = 0)
            embedding = process_text(annotation)
            embedding = tf.expand_dims(embedding, axis = 0)
            embeddings = tf.repeat(embedding, num_images , axis = 0)
        else:
            embeddings = ann_batch[:num_images]
        if ex_rate > 0:
             # 1.2 Преобразуем negative prompt в эмбеддинг
            negative_prompt = " "
            negative_prompt = tf.expand_dims(negative_prompt, axis = 0)
            negative_embedding = process_text(negative_prompt)
            negative_embedding = tf.expand_dims(negative_embedding, axis = 0)
            negative_embeddings = tf.repeat(negative_embedding, num_images , axis = 0)
            
        samples = tf.random.normal(shape=(num_images, image_size, image_size, img_channels), dtype=tf.float32) 
        bar = IntProgress(min=0, max=self.timesteps)
        display(bar)
        for t in reversed(range(0, self.timesteps)):
            bar.value+=1
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            if ema_mode:
                pred_noise = self.ema_network.predict([samples, tf.reshape(tt, shape = (-1, 1,1,1)), tf.reshape(embeddings, shape = (-1, 1,1,512)), small_images], verbose=0, batch_size=num_images)
            else:
                pred_noise = self.network.predict([samples, tf.reshape(tt, shape = (-1, 1,1,1)), tf.reshape(embeddings, shape = (-1, 1,1,512)), small_images], verbose=0, batch_size=num_images)
            if ex_rate >0:
                pred_negative_noise = self.ema_network.predict([samples, tf.reshape(tt, shape = (-1, 1,1,1)), tf.reshape(negative_embeddings, shape = (-1, 1,1,512)), small_images], verbose=0, batch_size=num_images)
                #Экстраполяция шума от negative в сторону positive
                resulted_noise = pred_noise + (pred_noise - pred_negative_noise)*ex_rate
            else:
                resulted_noise = pred_noise     
            samples = self.gdf_util.p_sample(resulted_noise, samples, tt, clip_denoised=True)
        
        generated_samples = samples
        self.plot_images(num_rows, num_cols, figsize, generated_samples*2.0)


total_timesteps = 300

widths = [[64, False], [128, False], [256, False], [512, False], [1024, True]]

block_depth = 3

batch_size = 64

img_channels = 3
clip_min = -1.0
clip_max = 1.0

image_size = 128


network = get_network(
    image_size=image_size,
    widths=widths,
    block_depth = block_depth
)
ema_network = get_network(
    image_size=image_size,
    widths=widths,
    block_depth = block_depth
)
ema_network.set_weights(network.get_weights())  # изначально веса равны

gdf_util = GaussianDiffusion(timesteps=total_timesteps)

model = DiffusionModel(network=network, ema_network=ema_network, gdf_util=gdf_util, timesteps=total_timesteps)
