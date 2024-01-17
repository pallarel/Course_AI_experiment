from runner import *
from dataset import *

if __name__ == '__main__':
    dataset = CatDogTestDataset(['data/train/cat', 'data/train/dog'], 256)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    model = ModelRunner(exp_name='test_exp', config_path='configs/catdog.yaml', device='cuda', mode='test')

    if(len(dataloader) > 20):
        print('Will only take the first 20 samples.')
    for i, sample in enumerate(dataloader):
        sample_image = sample['image'].cpu().numpy()[0]
        sample_label = sample['label'].cpu().numpy()[0]

        predicted_label = model.predict(sample_image)
        gt_label = np.argmax(sample_label)
        
        predicted_str = 'cat' if predicted_label == 0 else 'dog'
        gt_str = 'cat' if gt_label == 0 else 'dog'
        #print(f'gt: {sample_label}, predict: {predicted_label}')

        fig = plt.figure(dpi=150)
        ax = fig.add_subplot()
        ax.imshow(sample_image.transpose(1, 2, 0))
        plt.title(f'gt: {gt_str},  predict: {predicted_str}')
        plt.show()
        plt.close()

        #break
        if i >= 20:
            break

