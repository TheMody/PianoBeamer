import torch
import os
# Custom dataset class for loading images and masks

class piano_ds(torch.utils.data.Dataset):
    def __init__(self, path = "data/"):
        super(piano_ds, self).__init__()
        self.images = []
        self.masks = []
        for file in os.listdir(path + "images"):
            if file.endswith('.png'):
                # Assuming the mask has the same name as the image but in the masks directory
                self.images.append(os.path.join(path+ "images", file))
                self.masks.append(os.path.join(path+ "masks", file))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return image, mask
    

if __name__ == "__main__":
    dataset = piano_ds()
    print(f"Dataset length: {len(dataset)}")
    for i in range(len(dataset)):
        image, mask = dataset[i]

        #use matplotlib to display the image and mask
        import matplotlib.pyplot as plt
        img = plt.imread(image)
        msk = plt.imread(mask)

        img[:,:,0] = img[:,:,0] + msk*0.5 
        plt.imshow(img)
        plt.title("Image")
        plt.show()

