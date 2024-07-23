import cv2
import matplotlib.pyplot as plt
import albumentations as aug


def result_aug():

    IMAGE_SIZE = 256

    resize = aug.Resize(IMAGE_SIZE, IMAGE_SIZE)
    horizontal_flip = aug.HorizontalFlip(p=1.0)
    vertical_flip = aug.VerticalFlip(p=1.0)
    random_brightness_contrast = aug.RandomBrightnessContrast(p=1.0)

    image = cv2.imread('input/train/102122_sat.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized = resize(image=image)['image']
    h_flip = horizontal_flip(image=image)['image']
    v_flip = vertical_flip(image=image)['image']
    brightness_contrast = random_brightness_contrast(image=image)['image']

    all_augments = aug.Compose([
        aug.Resize(IMAGE_SIZE, IMAGE_SIZE),
        aug.HorizontalFlip(p=0.5),
        aug.VerticalFlip(p=0.5),
        aug.RandomBrightnessContrast(p=0.3)
    ])
    all_transformed = all_augments(image=image)['image']

    fig, ax = plt.subplots(1, 6, figsize=(20, 10))

    # 원본 이미지
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('on')

    # Resize 이미지
    ax[1].imshow(resized)
    ax[1].set_title('Resized Image')
    ax[1].axis('on')

    # Horizontal Flip 이미지
    ax[2].imshow(h_flip)
    ax[2].set_title('Horizontal Flip')
    ax[2].axis('on')

    # Vertical Flip 이미지
    ax[3].imshow(v_flip)
    ax[3].set_title('Vertical Flip')
    ax[3].axis('on')

    # Random Brightness Contrast 이미지
    ax[4].imshow(brightness_contrast)
    ax[4].set_title('Random Brightness Contrast')
    ax[4].axis('on')

    # All Augments 이미지
    ax[5].imshow(all_transformed)
    ax[5].set_title('All Augments')
    ax[5].axis('on')

    plt.savefig('result_aug.png')

    
def check_predict_time():
    df_img_path = pd.read_csv('random_data/test_split_3.csv')
    for idx, row in df_img_path.iterrows():
        img_path = row['IMAGES']
        
    
def main():
    result_aug()
    
if __name__ == '__main__':
    main()