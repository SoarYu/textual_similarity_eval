from SimCSEModel import *

model_path = '../pretrained_model/bert-base-chinese'

# 微调后参数存放位置
SAVE_PATH = 'saved_model/'

# 数据位置
# SNIL_TRAIN = './dataset/cnsd-snli/train.txt'
STS_TRAIN_LARGE = '../dataset/STS-B/sts-train.txt'
STS_TRAIN_BASE = '../dataset/STS-B/sts-train-base.txt'
STS_DEV = '../dataset/STS-B/sts-val.txt'
STS_TEST = '../dataset/STS-B/sts-test.txt'

if __name__ == '__main__':
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # load data
    train_data = load_data('sts', STS_TRAIN_BASE)
    random.shuffle(train_data)
    dev_data = load_data('sts', STS_DEV)
    test_data = load_data('sts', STS_TEST)
    train_dataloader = DataLoader(TrainDataset(tokenizer, train_data), batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(TestDataset(tokenizer, dev_data), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset(tokenizer, test_data), batch_size=BATCH_SIZE)

    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # train
    best = 0
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')

    # eval
    model = SimcseModel(pretrained_model=SAVE_PATH, pooling=POOLING)

    test_data = load_test_data(STS_TEST)
    srcs = []
    trgs = []
    labels = []
    for terms in test_data:
        src, trg, label = terms[0], terms[1], terms[2]
        srcs.append(src)
        trgs.append(trg)
        labels.append(label)
    logger.debug(f'{test_data[0]}')
    # sentence_embeddings = model.encode(srcs)
    # logger.debug(f"{type(sentence_embeddings)}, {sentence_embeddings.shape}, {sentence_embeddings[0].shape}")

    t1 = time.time()
    e1 = model.encode(srcs)
    e2 = model.encode(trgs)
    spend_time = time.time() - t1

    s = cos_sim(e1, e2)
    sims = []
    for i in range(len(srcs)):
        sims.append(s[i][i])
    sims = np.array(sims)
    labels = np.array(labels)
    spearman = compute_spearmanr(labels, sims)

    logger.debug(f'labels: {labels[:10]}')
    logger.debug(f'preds:  {sims[:10]}')
    logger.debug(f'Spearman: {spearman}')
    logger.debug(
        f'spend time: {spend_time:.4f}, count:{len(srcs + trgs)},spend_time:{spend_time}, qps: {len(srcs + trgs) / spend_time}')
    logger.debug(f'spearman: {spearman}')