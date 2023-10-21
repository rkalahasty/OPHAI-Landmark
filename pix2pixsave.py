if args.model_name == "pix2pix":
    BUFFER_SIZE = 400
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    # Each image is 256x256 in size
    IMG_WIDTH = 256
    IMG_HEIGHT = 256


    def resize(input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image


    def random_crop(input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

        return cropped_image[0], cropped_image[1]


    # Normalizing the images to [-1, 1]
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image


    @tf.function()
    def random_jitter(input_image, real_image):
        # Resizing to 286x286
        input_image, real_image = resize(input_image, real_image, 286, 286)

        # Random cropping back to 256x256
        input_image, real_image = random_crop(input_image, real_image)

        def jitter(input_image, real_image):
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
            return input_image, real_image

        def noJitter(input_image, real_image):
            return input_image, real_image

        input_image, real_image = tf.cond(tf.greater(tf.random.uniform(()), 0.5),
                                          lambda: jitter(input_image, real_image),
                                          lambda: noJitter(input_image, real_image))
        return input_image, real_image


    def load_image_train(image_file):
        input_image, real_image = image_file[0], image_file[1]
        input_image, real_image = random_jitter(input_image, real_image)
        input_image, real_image = normalize(input_image, real_image)

        return input_image, real_image


    train_dataset = tf.data.Dataset.from_tensor_slices([[i, j] for i, j in zip(X_train, y_train)])
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(len(X_train))
    train_dataset = train_dataset.batch(4)

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator = Pix2pix.Discriminator()
    generator = Pix2pix.Generator()


    @tf.function
    def train_step(input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = Pix2pix.generator_loss(disc_generated_output, gen_output,
                                                                               target)
            disc_loss = Pix2pix.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))


    def fit(train_ds, steps):
        start = time.time()
        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if (step) % 1000 == 0:
                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

                start = time.time()

                print(f"Step: {step // 1000}k")
            train_step(input_image, target, step)

            # Training step
            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)


    fit(train_dataset, steps=40000)
elif args.model_name == "dlib":
    print("Training Shape Predictor...")
    if os.path.basename(os.getcwd()) == "data_0":
        os.chdir("dlib_landmarks")
    dlib.train_shape_predictor("landmark_localization.xml", "landmark_predictor_dlib.dat", options)
else:
    os.chdir(args.sp)

    input_shape = (args.img, args.img, 3)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mean_absolute_error',
                      metrics=['mean_absolute_error', 'accuracy'])
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping

    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=8, callbacks=[es])

    os.chdir(args.sp)

    model.save(args.model_name + ".h5")

    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{args.model_name} Loss During Training')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower left')
    plt.savefig(f'{args.model_name}loss.png', bbox_inches='tight')
