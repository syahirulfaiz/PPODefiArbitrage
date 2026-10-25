#!/usr/bin/env python3
"""Render a new evaluation workbook, without frozen hashes or historical outcomes.
Run: python chart.py --input results/PPR01_V209_EVALUATION_job9851603.xlsx
No model training or test episodes are executed by this module.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = [
    "Greedy",
    "PPO",
    "SAC",
    "TD3",
    "Bellman_Ford_NetworkX",
    "Heuristic",
    "Beam_Search",
    "Grid_Search",
    "NLP_SciPy_Minimize",
]
DIMENSIONS = ["balance", "latency", "gas", "liquidity"]
LABELS = {
    "balance": "Wallet balance (USD)",
    #"latency": "Latency budget (seconds)",
    "latency": "Block-time budget (seconds)",
    #"gas": "Gas regime",
    "gas": "Gas price multiplier",
    "liquidity": "Liquidity depth",
}


def summarize(raw):
    keys = ["dimension", "ablation_value", "ablation_label", "method"]
    cells = raw.groupby(keys, as_index=False).agg(
        n_runs=("profit_usd", "size"),
        n_success=("success", "sum"),
        mean_profit_usd=("profit_usd", "mean"),
        std_profit_usd=("profit_usd", "std"),
        mean_roi_pct=("roi_pct", "mean"),
        std_roi_pct=("roi_pct", "std"),
        mean_gas_share_pct=("gas_cost_share_pct", "mean"),
        mean_atomic_time_seconds=("successful_atomic_complete_decision_time_seconds", "mean"),
        mean_cross_dex_pct=("cross_dex_share_pct", "mean"),
    )
    cells["reward_to_risk"] = cells.mean_roi_pct / cells.std_roi_pct.where(cells.std_roi_pct > 0)
    cells["success_rate"] = cells.n_success / cells.n_runs
    return cells


def load_runs(path):
    raw = (
        pd.read_csv(path)
        if path.suffix.lower() == ".csv"
        else pd.read_excel(path, sheet_name="raw_runs", engine="openpyxl")
    )
    required = {
        "dimension",
        "ablation_value",
        "ablation_label",
        "method",
        "seed",
        "profit_usd",
        "roi_pct",
        "gas_cost_share_pct",
        "success",
        "cross_dex_share_pct",
    }
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    keys = ["dimension", "ablation_value", "seed", "method"]
    if raw.duplicated(keys).any():
        raise ValueError("Duplicate scenario/method/seed rows.")
    if "successful_atomic_complete_decision_time_seconds" not in raw:

        def atomic_time(text):
            times = [t["complete_attempt_time_seconds"] for t in json.loads(text)]
            return float(np.mean(times)) if times else np.nan

        raw["successful_atomic_complete_decision_time_seconds"] = (
            raw.executed_transactions_json.map(atomic_time)
        )
    return raw


CELL_FONT_SIZE = 8.5
PPO_HIGHLIGHT = '#D62728'
DISPLAY = dict(zip(METHODS, ['Greedy','PPO','SAC','TD3','Bellman-Ford','Heuristic','Beam Search','Grid Search','NLP (SLSQP)']))
CMAPS = dict(zip(['mean_profit_usd','mean_roi_pct','reward_to_risk','mean_gas_share_pct','mean_cross_dex_pct','mean_atomic_time_seconds'], ['YlOrRd','YlGn','viridis','PuRd','cividis','PuBu']))
STYLES = dict(zip(METHODS, [('#0072B2','s'),('#3F3F3F','D'),('#9467BD','o'),('#8C564B','P'),('#CC79A7','+'),('#E69F00','v'),('#009E73','*'),('#D62728','^'),('#56B4E9','X')]))


def scenario_tick(dimension, value):
    if dimension == 'balance': return f'{value / 1000:g}K'
    if dimension == 'gas': return f'{value:g}x'
    return f'{value:g}'


def configure_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'font.size':10,'savefig.facecolor':'white','savefig.transparent':False})


def cell_text(value, column):
    if not np.isfinite(value): return 'N/A'
    if column == 'mean_profit_usd': return f'${value:,.0f}'
    if column == 'mean_cross_dex_pct': return f'{value:.0f}'
    if column == 'reward_to_risk': return f'{value:.1f}'
    if column == 'mean_atomic_time_seconds':
        if abs(value) >= 1000: return f'{value/1000:.1f}s'
        if abs(value) >= 100: return f'{value:.0f}'
        if abs(value) >= 10: return f'{value:.1f}'
    return f'{value:.2f}'


def atlas(cells, column, title, filename, output, formats, dpi, scale=1.0, log=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle
    methods = [m for m in METHODS if m in set(cells.method)]
    matrices, ticks = [], []
    for dimension in DIMENSIONS:
        sub = cells[cells.dimension == dimension]
        levels = sorted(sub.ablation_value.unique())
        matrices.append(sub.pivot(index='method',columns='ablation_value',values=column)
            .reindex(index=methods,columns=levels).to_numpy()*scale)
        ticks.append(levels)
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    positive = finite[finite > 0]
    if log and len(positive):
        lo,hi = float(positive.min()),float(positive.max())
        norm = LogNorm(lo,max(hi,lo*1.01))
    else:
        lo = min(0.,float(finite.min())) if len(finite) else 0.
        hi = float(finite.max()) if len(finite) else 1.
        norm = Normalize(lo,max(hi,lo+1e-12))
    if column == 'mean_cross_dex_pct': norm = Normalize(0,100)
    cmap = plt.get_cmap(CMAPS[column]).copy(); cmap.set_bad('#F2F2F2')
    fig,axes = plt.subplots(1,4,figsize=(14.65,4.08),
        gridspec_kw={'width_ratios':[max(1,len(t)) for t in ticks],'wspace':.075})
    fig.subplots_adjust(left=.105,right=.94,top=.86,bottom=.215)
    image = None
    for index,(ax,dimension,matrix,levels) in enumerate(zip(axes,DIMENSIONS,matrices,ticks)):
        if not levels:
            ax.set_axis_off(); continue
        shown = np.ma.masked_invalid(matrix)
        if log: shown = np.ma.masked_where(matrix <= 0,shown)
        image = ax.imshow(shown,cmap=cmap,norm=norm,aspect='auto',interpolation='none')
        ax.set_title(f'({chr(97+index)})',fontsize=12,fontweight='bold')
        ax.set_xticks(range(len(levels)),[scenario_tick(dimension,v) for v in levels],
            rotation=45,ha='right',rotation_mode='anchor')
        ax.set_yticks(range(len(methods)),[DISPLAY[m] for m in methods] if index == 0 else [])
        ax.set_xlabel(LABELS[dimension],fontsize=12,fontweight='bold')
        ax.set_xticks(np.arange(-.5,len(levels),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(methods),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.55,linestyle='--',alpha=.76)
        ax.tick_params(which='minor',bottom=False,left=False)
        if index: ax.tick_params(axis='y',length=0)
        for row in range(len(methods)):
            for col in range(len(levels)):
                value = matrix[row,col]; color = '#111111'
                if np.isfinite(value) and not (log and value <= 0):
                    r,g,b,_ = cmap(norm(value))
                    if .2126*r+.7152*g+.0722*b < .47: color = 'white'
                ax.text(col,row,cell_text(value,column),ha='center',va='center',
                    fontsize=CELL_FONT_SIZE,fontweight='bold',color=color,zorder=8)
        if 'PPO' in methods:
            row = methods.index('PPO')
            ax.add_patch(Rectangle((-.5,row-.5),len(levels),1,fill=False,
                edgecolor=PPO_HIGHLIGHT,linewidth=1.6,zorder=10,clip_on=False))
            if index == 0:
                tick = ax.get_yticklabels()[row]; tick.set_color(PPO_HIGHLIGHT); tick.set_fontweight('bold')
        ax.set_xlim(-.5,len(levels)-.5); ax.set_ylim(len(methods)-.5,-.5)
    if image is not None:
        bar = fig.colorbar(image,ax=axes,fraction=.022,pad=.01,aspect=25)
        bar.set_label(title+(' (log scale)' if log else ''),fontsize=10)
    for fmt in formats: fig.savefig(output/f'{filename}.{fmt}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def profit_time_scatter(overall, output, formats, dpi):
    """Render the existing overview/zoom and an additional standalone zoom PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    valid = overall[np.isfinite(overall.atomic_time)&(overall.atomic_time>0)&np.isfinite(overall.profit)].copy()
    valid['time_ms'] = valid.atomic_time*1000
    if len(valid):
        spread = float(valid.profit.max()-valid.profit.min())
        cluster = valid[valid.profit >= valid.profit.max()-max(spread*.15,1e-9)]
        if len(cluster)<2: cluster=valid
    else: cluster=valid
    offsets = {'PPO':(12,-18),'Greedy':(10,-28),'Heuristic':(-12,34),
        'Bellman_Ford_NetworkX':(-12,-38),'NLP_SciPy_Minimize':(12,16),
        'Beam_Search':(-16,-34),'Grid_Search':(12,34),'TD3':(12,20),'SAC':(12,-24)}

    def draw(ax, frame, title, annotate):
        ax.set_title(title,fontsize=12,fontweight='bold')
        ax.set_xscale('log')
        ax.set_xlabel('Mean successful atomic complete-decision time (ms, log scale)')
        ax.set_ylabel('Mean cumulative after-gas profit\nper episode (USD)')
        ax.grid(True,which='major',color='#D4DCE5',linestyle='--',linewidth=.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:,.0f}'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
        if frame.empty:
            ax.text(.5,.5,'No successful transactions: timing is N/A',transform=ax.transAxes,ha='center'); return
        for method in METHODS:
            if method not in frame.index: continue
            row=frame.loc[method]; color,marker=STYLES[method]
            ax.scatter(row.time_ms,row.profit,color=color,marker=marker,s=90 if method=='PPO' else 65,zorder=9)
            if method=='PPO':
                ax.scatter(row.time_ms,row.profit,marker='D',s=190,facecolors='none',edgecolors=PPO_HIGHLIGHT,linewidths=1.8,zorder=10)
            if annotate:
                dx,dy=offsets[method]
                ax.annotate(f'{DISPLAY[method]}\n{row.time_ms:.2f} ms, ${row.profit:,.2f}',
                    xy=(row.time_ms,row.profit),xytext=(dx,dy),textcoords='offset points',
                    ha='left' if dx>=0 else 'right',va='center',fontsize=9,
                    fontweight='bold' if method=='PPO' else 'normal',
                    arrowprops={'arrowstyle':'-','color':'#888888','lw':.8},
                    bbox={'boxstyle':'round,pad=.15','fc':'white','ec':'none','alpha':.85},zorder=12)
        xlo,xhi=frame.time_ms.min(),frame.time_ms.max(); ylo,yhi=frame.profit.min(),frame.profit.max()
        xp=max(np.log10(xhi/xlo)*.22,.07); yp=max((yhi-ylo)*.35,max(abs(yhi),1)*.0005)
        ax.set_xlim(xlo/10**xp,xhi*10**xp); ax.set_ylim(ylo-yp,yhi+yp)

    def decorate(fig, frame):
        handles=[Line2D([],[],color=STYLES[m][0],marker=STYLES[m][1],linestyle='none',markersize=7,label=DISPLAY[m]) for m in METHODS if m in frame.index]
        if handles: fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.095),ncol=4,frameon=False,fontsize=9)
        #note='Successful commits only; failed attempts and external network/block-inclusion time are excluded.'
        missing=[m for m in METHODS if m in overall.index and m not in valid.index]
        #if missing: note+='\n'+'; '.join(f"{DISPLAY[m]}: ${overall.loc[m,'profit']:,.2f} mean profit; successful-transaction time N/A" for m in missing)
        #fig.text(.5,.025,note,ha='center',va='bottom',fontsize=8,color='#555555')

    fig,axes=plt.subplots(1,2,figsize=(14.65,5.6))
    fig.subplots_adjust(left=.065,right=.985,bottom=.25,top=.89,wspace=.23)
    draw(axes[0],valid,'(a) All methods with successful transactions',False)
    draw(axes[1],cluster,'(b) Expanded high-profit cluster',True)
    decorate(fig,valid)
    for fmt in formats: fig.savefig(output/f'PPR01_Fig10_profit_time.{fmt}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    # Additional PNG only: the same cluster and coordinates, with no overview panel.
    fig,ax=plt.subplots(figsize=(8.5,5.6))
    fig.subplots_adjust(left=.13,right=.97,bottom=.25,top=.89)
    draw(ax,cluster,'Expanded high-profit cluster',True)
    decorate(fig,cluster)
    fig.savefig(output/'PPR01_Fig10_profit_time_expanded_cluster.png',dpi=dpi,bbox_inches='tight')
    plt.close(fig)


def pairwise_counts(raw):
    keys = ["dimension", "ablation_value", "seed"]
    profit = raw.pivot(index=keys, columns="method", values="profit_usd")
    times = raw.pivot(
        index=keys, columns="method", values="successful_atomic_complete_decision_time_seconds"
    )
    records = []
    for method in METHODS:
        if method == "PPO":
            continue
        delta = profit["PPO"] - profit[method]
        valid = times["PPO"].notna() & times[method].notna()
        time_delta = (times["PPO"] - times[method])[valid]
        wins, losses, ties = (
            int((delta > 1e-12).sum()),
            int((delta < -1e-12).sum()),
            int((delta.abs() <= 1e-12).sum()),
        )
        faster, slower, time_ties = (
            int((time_delta < -1e-12).sum()),
            int((time_delta > 1e-12).sum()),
            int((time_delta.abs() <= 1e-12).sum()),
        )
        records.append(
            {
                "comparator": method,
                "ppo_wins": wins,
                "ties": ties,
                "ppo_losses": losses,
                "ppo_faster": faster,
                "ppo_slower": slower,
                "time_ties": time_ties,
                "time_pairs_available": int(valid.sum()),
                "time_pairs_unavailable": int((~valid).sum()),
                "Cumulative profit (PPO / Other / Tie)": f"{wins} / {losses} / {ties}",
                "Successful atomic complete-decision time (PPO / Other / Tie)": (
                    f"{faster} / {slower} / {time_ties}" if valid.any() else "N/A"
                ),
            }
        )
    return pd.DataFrame(records)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Create seven figures and the paired PPO table.")
    parser.add_argument(
        "--input", type=Path, default=Path("results/PPR01_V209_EVALUATION_job9851603.xlsx")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=500)
    args = parser.parse_args()
    formats = args.formats.split(",")
    if set(formats) - {"png", "pdf"} or args.dpi < 1:
        parser.error("Use png,pdf and a positive DPI.")
    configure_style()
    raw = load_runs(args.input)
    cells = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output_dir / "atlas_cells.csv", index=False)
    specs = [
        (
            "mean_profit_usd",
            "Mean simulated after-gas profit (USD)",
            "PPR01_Fig04_mean_profit_performance_atlas",
            1,
            False,
        ),
        ("mean_roi_pct", "Mean simulated ROI (%)", "PPR01_Fig05_mean_roi_performance_atlas", 1, False),
        (
            "reward_to_risk",
            "Simulated reward-to-risk ratio",
            "PPR01_Fig06_simulated_reward_to_risk_performance_atlas",
            1,
            True,
        ),
        (
            "mean_gas_share_pct",
            "Gas share (%)",
            "PPR01_Fig07_gas_share_performance_atlas",
            1,
            True,
        ),
        (
            "mean_cross_dex_pct",
            "Cross-DEX arbitrage share (%)",
            "PPR01_arbitrage_type_heat_1x4",
            1,
            False,
        ),
        (
            "mean_atomic_time_seconds",
            "Successful atomic complete-decision time (ms)",
            "PPR01_Fig09_inference_time_performance_atlas",
            1000,
            True,
        ),
    ]
    for column, title, filename, scale, log in specs:
        atlas(cells, column, title, filename, args.output_dir, formats, args.dpi, scale, log)
    overall = raw.groupby("method").agg(
        profit=("profit_usd", "mean"),
        atomic_time=("successful_atomic_complete_decision_time_seconds", "mean"),
    )
    profit_time_scatter(overall, args.output_dir, formats, args.dpi)
    # Count real wins, ties and losses; do not impose PPO dominance.

    pairwise_counts(raw).to_csv(
        args.output_dir / "PPR01_Table04_ppo_pairwise_counts.csv", index=False
    )
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
