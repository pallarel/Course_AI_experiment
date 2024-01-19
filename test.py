from runner import ModelRunner
from runner_nl import ModelRunnerNL
from dataset import *



if __name__ == '__main__':

    def _catdog_test():
        dataset = CatDogTestDataset(['data/catdog/train/cat', 'data/catdog/train/dog'], 256)
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        model = ModelRunner(exp_name='effnet_pretrained', config_path='configs/catdog/catdog_another.yaml', device='cuda', mode='test')

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
    
    def _chinese_test():
        dataset = ChineseTitleDataset(['data/chinese/test.xlsx'])
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        model = ModelRunnerNL(exp_name='chinese_2', config_path='configs/chinese/chinese.yaml', device='cuda', mode='test')

        #if(len(dataloader) > 20):
        #    print('Will only take the first 20 samples.')
        for i, sample in enumerate(dataloader):
            sample_tokens = sample['tokens']
            sample_mask = sample['mask']
            sample_str = sample['seq']
            sample_label = sample['label'].cpu().numpy()[0]

            predicted_label = model.predict(sample_tokens, sample_mask)
            gt_label = np.argmax(sample_label)
           
            print(f'str: {sample_str}, gt: {gt_label}, predict: {predicted_label}')

            #break
            #if i >= 20:
            #    break

    def _chinese_single_test():
        C_tokenizer = ChineseTitleTokenizer()
        model = ModelRunnerNL(exp_name='chinese_2', config_path='configs/chinese/chinese.yaml', device='cuda', mode='test')

        while(True):
            str = input('str: ')
            if str == 'q':
                break
            
            tokens, mask = C_tokenizer.process(str)
            predicted_label = model.predict(tokens.unsqueeze(dim=0), mask.unsqueeze(dim=0))

            print(f'predict: {predicted_label}')



    _chinese_test()


