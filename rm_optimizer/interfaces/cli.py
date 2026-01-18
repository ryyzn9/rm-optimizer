"""
Command-line interface for RM-Optimizer.

Usage:
    rm-opt train --model deberta --optimizer muon --epochs 5
    rm-opt analyze --checkpoint model.ckpt --type hessian
    rm-opt compare --optimizers adam,muon,lion --seeds 3
"""

import typer
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="rm-opt",
    help="RM-Optimizer: Reward Model Analysis Framework",
    add_completion=False
)

console = Console()


@app.command()
def train(
    model: str = typer.Option(
        "microsoft/deberta-v3-large",
        "--model", "-m",
        help="HuggingFace model identifier"
    ),
    optimizer: str = typer.Option(
        "adamw",
        "--optimizer", "-o",
        help="Optimizer: adam, adamw, sgd, muon, lion"
    ),
    data_path: Optional[str] = typer.Option(
        None,
        "--data", "-d",
        help="Path to preference data (parquet/json)"
    ),
    epochs: int = typer.Option(
        5,
        "--epochs", "-e",
        help="Number of training epochs"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size", "-b",
        help="Training batch size"
    ),
    learning_rate: float = typer.Option(
        1e-5,
        "--lr",
        help="Learning rate"
    ),
    checkpoint_dir: str = typer.Option(
        "checkpoints",
        "--checkpoint-dir",
        help="Directory for saving checkpoints"
    ),
    wandb: bool = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Enable Weights & Biases logging"
    ),
    precision: str = typer.Option(
        "bf16-mixed",
        "--precision",
        help="Training precision (32, 16, bf16-mixed)"
    )
):
    """Train a reward model."""
    console.print(f"\n[bold blue]RM-Optimizer Training[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Optimizer: {optimizer}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print()
    
    if data_path is None:
        console.print("[red]Error: --data is required[/red]")
        raise typer.Exit(1)
    
    from rm_optimizer.training.lightning_module import train_reward_model
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Training reward model...", total=None)
        
        checkpoint_path = train_reward_model(
            model_name=model,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=epochs,
            data_path=data_path,
            checkpoint_dir=checkpoint_dir,
            use_wandb=wandb,
            precision=precision
        )
        
        progress.update(task, completed=True)
    
    console.print(f"\n[green]✓ Training complete![/green]")
    console.print(f"Checkpoint saved to: {checkpoint_path}")


@app.command()
def analyze(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint", "-c",
        help="Path to model checkpoint"
    ),
    analysis_type: str = typer.Option(
        "hessian",
        "--type", "-t",
        help="Analysis type: hessian, calibration, rl-readiness"
    ),
    data_path: Optional[str] = typer.Option(
        None,
        "--data", "-d",
        help="Path to preference data for analysis"
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        help="Number of top eigenvalues to compute"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results"
    )
):
    """Analyze a trained reward model."""
    console.print(f"\n[bold blue]RM-Optimizer Analysis[/bold blue]")
    console.print(f"Checkpoint: {checkpoint}")
    console.print(f"Analysis: {analysis_type}")
    console.print()
    
    from rm_optimizer.core.reward_model import BradleyTerryRM
    
    # Load model
    with console.status("Loading model..."):
        model = BradleyTerryRM.from_pretrained(checkpoint)
    
    if analysis_type == "hessian":
        from rm_optimizer.landscape.hessian import HessianAnalyzer
        from rm_optimizer.core.data_loader import create_dataloader
        
        if data_path is None:
            console.print("[red]Error: --data required for Hessian analysis[/red]")
            raise typer.Exit(1)
        
        loader = create_dataloader(
            data_path,
            tokenizer=model.tokenizer,
            batch_size=8,
            shuffle=False
        )
        
        with console.status("Computing Hessian eigenspectrum..."):
            analyzer = HessianAnalyzer(model)
            spectrum = analyzer.compute_spectrum(loader, top_k=top_k)
        
        # Display results
        table = Table(title="Hessian Analysis Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Top Eigenvalue", f"{spectrum.eigenvalues[0]:.4f}")
        table.add_row("Trace", f"{spectrum.trace_estimate:.4f}")
        table.add_row("Condition Number", f"{spectrum.condition_number:.4f}")
        table.add_row("Effective Rank", f"{spectrum.effective_rank:.4f}")
        table.add_row("Sharpness", f"{spectrum.sharpness_score:.6f}")
        table.add_row("Flatness", f"{spectrum.flatness_index:.6f}")
        
        console.print(table)
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(spectrum.to_dict(), f, indent=2)
            console.print(f"\nResults saved to: {output}")
    
    elif analysis_type == "rl-readiness":
        from rm_optimizer.rl_coupling.rl_readiness import RLReadinessScorer
        
        scorer = RLReadinessScorer()
        # Use default values for demo
        report = scorer.compute_score(
            accuracy=0.85,
            ece=0.08,
            reward_variance=1.2,
            top_eigenvalue=150.0
        )
        
        console.print(str(report))
    
    else:
        console.print(f"[red]Unknown analysis type: {analysis_type}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    optimizers: str = typer.Option(
        "adam,adamw,muon",
        "--optimizers", "-o",
        help="Comma-separated list of optimizers"
    ),
    model: str = typer.Option(
        "microsoft/deberta-v3-large",
        "--model", "-m",
        help="Model to train"
    ),
    data_path: Optional[str] = typer.Option(
        None,
        "--data", "-d",
        help="Path to preference data"
    ),
    seeds: int = typer.Option(
        3,
        "--seeds", "-s",
        help="Number of random seeds per optimizer"
    ),
    epochs: int = typer.Option(
        5,
        "--epochs", "-e",
        help="Training epochs"
    ),
    output_dir: str = typer.Option(
        "outputs/optimizer_comparison",
        "--output-dir",
        help="Output directory"
    )
):
    """Compare multiple optimizers."""
    optimizer_list = [o.strip() for o in optimizers.split(",")]
    
    console.print(f"\n[bold blue]RM-Optimizer Comparison[/bold blue]")
    console.print(f"Optimizers: {optimizer_list}")
    console.print(f"Seeds: {seeds}")
    console.print()
    
    if data_path is None:
        console.print("[red]Error: --data is required[/red]")
        raise typer.Exit(1)
    
    from rm_optimizer.landscape.optimizer_comparison import OptimizerComparison
    
    comparison = OptimizerComparison(
        model_name=model,
        data_path=data_path,
        optimizers=optimizer_list,
        num_seeds=seeds,
        epochs=epochs,
        output_dir=output_dir
    )
    
    comparison.run_experiment()
    
    # Generate and display report
    df = comparison.generate_report()
    console.print(f"\n[green]✓ Comparison complete![/green]")
    console.print(f"Results saved to: {output_dir}")


@app.command()
def info():
    """Display system and package information."""
    import torch
    import platform
    
    console.print("\n[bold blue]RM-Optimizer System Info[/bold blue]\n")
    
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")
    
    # System info
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    
    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        table.add_row("BF16 Support", str(torch.cuda.is_bf16_supported()))
    
    console.print(table)


@app.command()
def download(
    dataset: str = typer.Argument(
        "hh-rlhf",
        help="Dataset name: hh-rlhf, ultrafeedback, shp, helpsteer"
    ),
    output_dir: str = typer.Option(
        "data",
        "--output", "-o",
        help="Output directory"
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples", "-n",
        help="Maximum samples to download (None for all)"
    ),
    format: str = typer.Option(
        "parquet",
        "--format", "-f",
        help="Output format: parquet, json, csv"
    )
):
    """Download a preference dataset."""
    from rm_optimizer.core.datasets import download_dataset, list_datasets
    
    if dataset == "list":
        list_datasets()
        return
    
    console.print(f"\n[bold blue]Downloading {dataset}...[/bold blue]")
    
    path = download_dataset(
        name=dataset,
        output_dir=output_dir,
        max_samples=max_samples,
        format=format
    )
    
    console.print(f"\n[green]✓ Dataset saved to: {path}[/green]")


@app.callback()
def main():
    """
    RM-Optimizer: A framework for reward model analysis.
    
    Analyze loss landscape geometry, RL coupling dynamics,
    and data efficiency for reward models.
    """
    pass


if __name__ == "__main__":
    app()
