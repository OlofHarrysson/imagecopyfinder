import progressbar as pbar

class ProgressbarWrapper():
  def __init__(self, n_epochs, n_batches):
    self.text = pbar.FormatCustomText(
      'Epoch: %(epoch).d/%(n_epochs).d, Batch: %(batch)d/%(n_batches)d',
      dict(
        epoch=0,
        n_epochs=n_epochs,
        batch=0,
        n_batches=n_batches
      ),
    )

    self.bar = pbar.ProgressBar(widgets=[
        pbar.Percentage(), ' ',
        self.text, ' ',
        pbar.Bar(), ' ',
        pbar.Timer(), ' ',
        pbar.AdaptiveETA(), ' ',
    ], redirect_stdout=True)

    self.bar.start()

  def __call__(self, down_you_go):
    return self.bar(down_you_go)

  def update(self, epoch, batch):
    self.text.update_mapping(epoch=epoch, batch=batch)
    self.bar.update()