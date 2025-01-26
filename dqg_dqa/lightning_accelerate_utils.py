import os, numpy as np, re, shutil

def accelerate_forward(args, model, monitor = 'val_loss', save_model = True):
    track_metric = []
    track_ckpts  = {}
    # do training and validation
    start_epoch = 0
    if args.resume_ckpt_path is not None:
        start_epoch = int(re.search(r'(?:epoch)(\d+)', args.resume_ckpt_path).group(1))
    args.max_epochs = 0 if getattr(args, 'test_only', False) else args.max_epochs
    c_do_save_state = getattr(args, 'test_only', False) == False and not getattr(args, 'do_lora', False)
    for epoch in range(start_epoch, args.max_epochs):
        print('🟧🟧Training for epoch:', epoch)
        model.train()
        model.to(args.device)
        model.do_acc_train(epoch)
        losses = model.train_losses[epoch]
        model.accelerator.log({"train_loss": np.mean(losses), "epoch": epoch}, step = epoch)

        if getattr(args, 'do_validation', True):
            print('🟦🟧Validating for epoch:', epoch)
            model.eval()
            model.to(args.device)
            model.do_acc_validate(epoch)
            if monitor in ['val_loss']:
                track_metric_epoch = np.mean(model.val_losses[epoch])
                model.accelerator.log({monitor: track_metric_epoch, "epoch": epoch}, step = epoch)

                # check for early stopping (min)
                track_metric.append(track_metric_epoch)
                if len(track_metric) > args.patience:
                    cut = track_metric[-args.patience:]
                    if all([cut[i+1] >= cut[i] for i in range(args.patience-1)]):
                        print(f'🟥🟥Early stopping at epoch {epoch} as {monitor} did not decrease')
                        break
                
            elif monitor in ['eval_em', 'eval_bleu']:
                track_metric_epoch = getattr(model, f'{monitor}s')[epoch] # value is single float
                model.accelerator.log({monitor: track_metric_epoch, "epoch": epoch}, step = epoch)

                # check for early stopping (max)
                track_metric.append(track_metric_epoch)
                if len(track_metric) > args.patience:
                    cut = track_metric[-args.patience:]
                    if all([cut[i+1] <= cut[i] for i in range(args.patience-1)]):
                        print(f'🟥🟥Early stopping at epoch {epoch} as {monitor} did not increase')
                        break

            # save the model
            fp_ckpt = args.save_dir + f'/model_epoch{epoch}.ckpt'
            if c_do_save_state and save_model:
                model.accelerator.save_state(output_dir = fp_ckpt, safe_serialization = False)
                track_ckpts[epoch] = fp_ckpt

            print('🧹Cleaning up the saved checkpoints')  
            saves = sorted(track_metric)[:args.save_top_k_models]
            deletes = [epoch for epoch, x in enumerate(track_metric) if x not in saves]
            for epoch, fp_ckpt in track_ckpts.items():
                if epoch in deletes and os.path.exists(fp_ckpt): 
                    shutil.rmtree(fp_ckpt)
                    print(f'\t🗑🗑Housekeeping - deleted checkpoint: {fp_ckpt}')
            
    model.accelerator.end_training()
    # do testing
    print('🟦🟦TESTING MODEL...')
    if getattr(args, 'do_test', True): 
        model.eval()
        model.to(args.device)
        model.do_acc_test()
    print('🟩🟩MODEL TESTED!')
    if c_do_save_state and save_model:
        best_epoch = np.argmin(track_metric)
        best_ckpt  = track_ckpts[best_epoch]
        model.accelerator.load_state(best_ckpt)

        model.accelerator.save_state(output_dir = args.save_dir + f'/tested_model_epoch{epoch}.ckpt', 
                                safe_serialization = False)
        shutil.rmtree(best_ckpt) # delete the ckpt (save as "tested*")

    print('🏁🏁Done!')
    return model

def lightning_forward(args, model, trainer = None):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping 
    from lightning.pytorch.callbacks.progress import TQDMProgressBar
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.strategies import DDPStrategy
    
    if trainer == None:
        # set up trainer 
        monitor, mode = 'val_loss', 'min'
        callbacks = [
                    EarlyStopping(monitor = monitor, mode = mode, check_finite = True,
                    min_delta = 0.00, patience = args.patience, check_on_train_epoch_end = False),
                    # do not set stopping_threshold = 0.0, # log_prob 0 or more
                    TQDMProgressBar(refresh_rate = 20), 
                    LearningRateMonitor(logging_interval = 'step'),
                    ModelCheckpoint(dirpath = args.save_dir, monitor = monitor, mode = mode,  
                                    every_n_epochs = 1, save_top_k = 2, save_on_train_epoch_end = True)
                    ]
        precision = '32-true' if args.fp32 else 'bf16-mixed'
        strategy  = DDPStrategy(find_unused_parameters = True if not args.do_baseline else False)
        trainer   = pl.Trainer(callbacks = callbacks, devices = 'auto', strategy = strategy, 
                            precision = precision, max_epochs = args.max_epochs,)
        print('TRAINER SET UP')

    # train model
    if not getattr(args, 'test_only', False) and getattr(args, 'do_test', False):
        trainer.fit(model, ckpt_path = args.resume_ckpt_path)
    print('MODEL TRAINED and saved to', args.save_dir)

    # test model
    print('🟦🟦TESTING MODEL...')
    trainer.test(model)
    print('🟩🟩MODEL TESTED')