"""Train/evaluate multiple ADC precisions using one leakage-free dataset."""
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .simulation import SimulationConfig, make_splits, denormalize_target
from .model import HybridLocalizer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader(split: dict, batch_size: int, shuffle: bool = False):
    ds = TensorDataset(torch.from_numpy(split["x"]),torch.from_numpy(split["y"]))
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=0)


def train_model(model, splits, *, epochs, learning_rate, batch_size, device):
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    train_data=loader(splits["train"],batch_size,True)
    val_data=loader(splits["val"],batch_size,False)
    best_val = float('inf')
    history = []
    best_weights = None
    for epoch in range(1,epochs+1):
        model.train()
        losses=[]
        for x,y in train_data:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            out=model(x)
            loss=loss_fn(out,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            opt.step()
            losses.append(loss.item()*len(x))
        model.eval()
        with torch.no_grad():
            val_errors=[]
            for x,y in val_data:
                p=model(x.to(device))
                val_errors.append(loss_fn(p,y.to(device)).item()*len(x))
        val_loss=sum(val_errors)/len(splits['val']['x'])
        train_loss=sum(losses)/len(splits['train']['x'])
        history.append({"epoch":epoch,"train_mse":train_loss,"val_mse":val_loss})
        print(f"  epoch {epoch:02d}: train MSE={train_loss:.5f} val MSE={val_loss:.5f}",flush=True)
        if val_loss<best_val:
            best_val=val_loss
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    return history


def test_model(model, split, radius_range, device, batch_size):
    model.eval()
    rows=[]
    snrs=np.unique(split['snr_db'])
    with torch.no_grad():
        for snr in snrs:
            mask=(split['snr_db']==snr)
            pred=[]
            clip=[]
            data=split['x'][mask]
            for i in range(0,len(data),batch_size):
                x=torch.from_numpy(data[i:i+batch_size]).to(device)
                estimates,rate=model.forward_with_diagnostics(x)
                pred.append(estimates.cpu().numpy())
                clip.append((float(rate.cpu()),len(x)))
            predicted=np.concatenate(pred,axis=0)
            expected=split['y'][mask]
            cart_pred=denormalize_target(predicted,radius_range)
            cart_true=denormalize_target(expected,radius_range)
            pos_diff=np.linalg.norm(cart_pred-cart_true,axis=1)
            rmse=float(np.sqrt(np.mean(pos_diff**2)))
            mae=float(np.mean(pos_diff))
            rows.append({"snr_db":float(snr),"position_rmse_m":rmse,
                         "position_mae_m":mae,"clip_fraction":sum(r*n for r,n in clip)/len(data),
                         "test_samples":len(data)})
    return rows


def save_csv(path, rows):
    if not rows:
        return
    with open(path,'w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--antennas',type=int,default=64)
    parser.add_argument('--rf-chains',type=int,default=8)
    parser.add_argument('--carrier-ghz',type=float,default=25)
    parser.add_argument('--snrs',type=float,nargs='+',default=[-15,-10,-5,0,5,10,15,20])
    parser.add_argument('--samples-per-snr',type=int,default=300)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--full-scale',type=float,default=3.0)
    parser.add_argument('--architectures',nargs='+',choices=['cnn','mlp'],default=['cnn','mlp'])
    parser.add_argument('--bit-depths',nargs='+',choices=['float','1','2','4'],default=['float','1','2','4'])
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--output-dir',type=Path,default=Path('results'))
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if args.antennas % args.rf_chains or args.rf_chains < 1:
        parser.error('antennas must be divisible by rf-chains')
    cfg=SimulationConfig(antennas=args.antennas,carrier_hz=args.carrier_ghz*1e9,
                         snr_db=tuple(args.snrs),samples_per_snr=args.samples_per_snr,seed=args.seed)
    splits=make_splits(cfg)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    print('Device:', device,'Split sizes:',{k:len(v['x']) for k,v in splits.items()},flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with open(args.output_dir/'config.json','w',encoding='utf-8') as fh:
        json.dump({**vars(args),'output_dir':str(args.output_dir),'device':str(device),
                   'license_note':'Original inspiration: Fabiani et al. ICC 2025; see ATTRIBUTION.md'},fh,indent=2)
    result_rows=[]; histories=[]
    for arch in args.architectures:
        for bits in args.bit_depths:
            print(f'Running {arch}, {bits}-bit ADC',flush=True)
            # Same initial seed for fair comparisons; architectures differ.
            seed_everything(args.seed)
            model=HybridLocalizer(args.antennas,args.rf_chains,arch,
                                  None if bits=='float' else int(bits),args.full_scale).to(device)
            hist=train_model(model,splits,epochs=args.epochs,learning_rate=args.learning_rate,
                             batch_size=args.batch_size,device=device)
            rows=test_model(model,splits['test'],(cfg.range_min_m,cfg.range_max_m),device,args.batch_size)
            for row in rows:
                result_rows.append({'architecture':arch,'bits':bits,**row})
            for row in hist:
                histories.append({'architecture':arch,'bits':bits,**row})
            torch.save(model.state_dict(),args.output_dir/f'weights_{arch}_{bits}.pt')
    save_csv(args.output_dir/'metrics.csv',result_rows)
    save_csv(args.output_dir/'training_history.csv',histories)
    from .plotting import plot_results
    plot_results(result_rows,args.output_dir)
    print('Finished. Metrics:',args.output_dir/'metrics.csv',flush=True)


if __name__=='__main__':
    main()
