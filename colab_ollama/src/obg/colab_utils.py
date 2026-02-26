@app.command("sync-ssh")
def sync_ssh(
    src: Path = typer.Option(
        Path("/content/drive/MyDrive/Colab Notebooks/.ssh"),
        "--src",
        help="Source .ssh directory (Google Drive mounted path)",
    ),
    dst: Path = typer.Option(
        Path("/content/.ssh"),
        "--dst",
        help="Destination .ssh directory on Colab VM",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing destination by deleting it first",
    ),
):
    """Copy .ssh from Drive into /content and set safe permissions."""
    if not src.exists() or not src.is_dir():
        raise typer.BadParameter(f"Source does not exist or is not a directory: {src}")

    if dst.exists():
        if not force:
            raise typer.BadParameter(f"Destination already exists: {dst}. Use --force.")
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    os.chmod(dst, 0o700)
    for p in dst.rglob("*"):
        os.chmod(p, 0o700 if p.is_dir() else 0o600)

    typer.echo(f"✅ Copied {src} -> {dst} and set permissions.")
