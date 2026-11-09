"""Offline official-template formatter; never uploads or manufactures predictions."""
import argparse
import re
from pathlib import Path
import pandas as pd


def format_predictions(template_path,prediction_path,output_path):
    output_path=Path(output_path)
    if output_path.exists():raise FileExistsError(output_path)
    # Read the template as strings to preserve stock IDs and its exact order/header.
    template=pd.read_csv(template_path,dtype=str)
    if len(template)!=50 or len(template.columns)<2:raise ValueError('Expected 50-stock wide official template')
    key=template.columns[0]
    if template[key].isna().any() or template[key].duplicated().any():raise ValueError('Invalid template stock IDs')
    dates=list(template.columns[1:])
    if not all(re.fullmatch(r'\d{4}-\d{2}-\d{2}',x) for x in dates):raise ValueError('Expected ISO date headers')
    pred=pd.read_csv(prediction_path,dtype={'stock_id':str})
    if not {'date','stock_id','pred_label'}<=set(pred):raise ValueError('Missing prediction columns')
    if pred.duplicated(['stock_id','date']).any():raise ValueError('Duplicate stock/date predictions')
    if not pred.pred_label.isin([0,1]).all():raise ValueError('Predictions must be binary')
    required=pd.MultiIndex.from_product([template[key].tolist(),dates],names=['stock_id','date'])
    indexed=pred.set_index(['stock_id','date']).pred_label
    missing=required.difference(indexed.index)
    if len(missing):raise ValueError(f'Missing {len(missing)} required predictions; never fill placeholders')
    for date in dates:
        template[date]=[int(indexed.loc[(stock,date)]) for stock in template[key]]
    template.to_csv(output_path,index=False)
    readback=pd.read_csv(output_path,dtype={key:str})
    assert readback[key].tolist()==template[key].tolist() and readback.columns.tolist()==template.columns.tolist()
    assert readback[dates].isin([0,1]).all().all()
    return output_path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(format_predictions(a.template,a.predictions,a.output))
