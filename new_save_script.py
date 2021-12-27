
# the issue is that with our previous saving script the config file did not report any information about the model state
# hence, I need to adopt and refactor this new saving function to our script

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
   torch.save(state, filename)
if is_best:
   shutil.copyfile(filename, 'model_best.pth.tar')

save_checkpoint({
   'epoch': epoch + 1,
   'arch': args.arch,
   'state_dict': model.state_dict(),
   'best_prec1': best_prec1,
   'optimizer': optimizer.state_dict()
}, is_best)