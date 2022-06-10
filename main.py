from util import *
from model import *

def train(args, model, optimizer, train_loader, vali_loader, scheduler):

    # Loss Function
    criterion = nn.L1Loss().to(args.device)
    best_mae = 9999
    
    for epoch in range(1,args.epochs+1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(args.device), label.float().to(args.device)
            
            optimizer.zero_grad()

            # Data -> Model -> Output
            logit = model(img)
            # Calc loss
            loss = criterion(logit.squeeze(1), label)

            # backpropagation
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            
        if scheduler is not None:
            scheduler.step()
            
        # Evaluation Validation set
        vali_mae = validation(model, vali_loader, criterion, args.device)
        
        print(f'Epoch [{epoch}] Train MAE : [{np.mean(train_loss):.5f}] Validation MAE : [{vali_mae:.5f}]\n')
        
        # Model Saved
        if best_mae > vali_mae:
            best_mae = vali_mae
            torch.save(model.state_dict(), './saved/best_model.pth')
            print('Model Saved.')

def validation(model, vali_loader, criterion, device):
    model.eval() # Evaluation
    vali_loss = []
    with torch.no_grad():
        for img, label in tqdm(iter(vali_loader)):
            img, label = img.float().to(device), label.float().to(device)

            logit = model(img)
            loss = criterion(logit.squeeze(1), label)
            
            vali_loss.append(loss.item())

    vali_mae_loss = np.mean(vali_loss)
    return vali_mae_loss

def evaluate(model, test_loader, device):
    model.eval()
    result = list()
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)

            logit = model(img)
            predict = logit.squeeze(1).detach().cpu().tolist()

            result += predict            
    
    print(result)
    f = open("result.csv", 'w')
    for i in range(len(result)):
        data = "%d\n" % result[i]
        f.write(data)
    f.close()
    print('test done, saved')
    

def main():
    print('start')
    parser = default_parser(argparse.ArgumentParser())
    args = parser.parse_args()

    print("===== [ 모델 학습 정보 ({}) ] =====".format(time.time()))

    # seed 선언
    seed_everything(args.seed) 

    # GPU 설정  
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 데이터 로드
    all_img_path, all_label = get_train_data('./dataset/train')
    test_img_path = get_test_data('./dataset/test')
    print('data loaded')

    # Train : Validation = 0.8 : 0.2 Split
    train_len = int(len(all_img_path)*0.8)
    train_img_path = all_img_path[:train_len]
    train_label = all_label[:train_len]
    vali_img_path = all_img_path[train_len:]
    vali_label = all_label[train_len:]

    #train_img_path = all_img_path[:10]
    #train_label = all_label[:10]
    #vali_img_path = all_img_path[10:20]
    #vali_label = all_label[10:20]
    print('train len : ', len(train_label), 'validation len : ', len(vali_label))

    # data transform
    train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])

    test_transform = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Resize((args.img_size, args.img_size)),
                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                   ])

    # Get Dataloader
    train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)

    vali_dataset = CustomDataset(vali_img_path, vali_label, train_mode=True, transforms=test_transform)
    vali_loader = DataLoader(vali_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)

    test_label = [0 for _ in range(len(test_img_path))]
    test_dataset = CustomDataset(test_img_path, test_label, train_mode=False, transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    
    # define model
    model = CNNRegressor()
    model.to(args.device)    
    print('model : CNNRegressor')    

    #train
    optimizer = torch.optim.SGD(params = model.parameters(), lr = args.learning_rate)
    scheduler = None

    train(args, model, optimizer, train_loader, vali_loader, scheduler)    
    evaluate(model, test_loader, args.device)


if __name__ == '__main__':
    main()
  