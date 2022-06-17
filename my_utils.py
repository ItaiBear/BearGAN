import pickle
from types import SimpleNamespace
from constants import root_project_directory
import utils.utils as utils
import os

def load_options(opt):
    file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt

def get_default_opt(train):
  #do not change these parameters
  opt = SimpleNamespace(
      #--- general options ---gdfsgds
      name="oasis_cityscapes_pretrained",                                     # name of the experiment. It decides where to store samples and models
      seed=42,                                                                # random seed
      segmentation="semantic",                                                # use semantic or panoptic segmentation for training
      gpu_ids='0',                                                            # gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU
      checkpoints_dir=root_project_directory+"/pretrained_checkpoints",       # models are saved here
      no_spectral_norm=False,                                                 # this option deactivates spectral norm in all layers
      batch_size=20,                                                           # input batch size
      dataroot=root_project_directory+"/dataset",                 # path to dataset root
      dataset_mode="cityscapes",                                              # this option indicates which dataset should be loaded
      no_flip=False,                                                          # if specified, do not flip the images for data argumentation
      #--- generator options ---
      num_res_blocks=6,                                                       # number of residual blocks in G and D
      channels_G=64,                                                          # number of gen filters in first conv layer in generator
      param_free_norm="syncbatch",                                            # which norm to use in generator before SPADE
      spade_ks=3,                                                             # kernel size of convs inside SPADE
      no_EMA=False,                                                           # if specified, do *not* compute exponential moving averages
      EMA_decay=0.9999,                                                       # decay in exponential moving averages
      no_3dnoise=False,                                                       # if specified, do *not* concatenate noise to label maps
      z_dim=64)                                                               # dimension of the latent z vector
  
  if train:
    opt_extra = SimpleNamespace(
        freq_print=1000,                                                      # frequency of showing training results
        freq_save_ckpt=20000,                                                 # frequency of saving the checkpoints
        freq_save_latest=10000,                                               # frequency of saving the latest model
        freq_smooth_loss=250,                                                 # smoothing window for loss visualization
        freq_save_loss=2500,                                                  # frequency of loss plot updates
        freq_fid=5000,                                                        # frequency of saving the fid score (in training iterations)
        continue_train=True,                                                 # resume previously interrupted training
        which_iter='latest',                                                  # which epoch to load when continue_train
        num_epochs=10,                                                       # number of epochs to train
        beta1=0.0,                                                            # momentum term of adam
        beta2=0.999,                                                          # momentum term of adam
        lr_g=0.0001,                                                          # G learning rate
        lr_d=0.0004,                                                          # D learning rate

        channels_D=64,                                                        # number of discrim filters in first conv layer in discriminator
        add_vgg_loss=False,                                                   # if specified, add VGG feature matching loss
        lambda_vgg=10.0,                                                      # weight for VGG loss
        no_balancing_inloss=False,                                            # if specified, do *not* use class balancing in the loss function
        no_labelmix=False,                                                    # if specified, do *not* use LabelMix
        lambda_labelmix=10.0                                                  # weight for LabelMix regularization
    )
  else:
    opt_extra = SimpleNamespace(
        results_dir=root_project_directory+"/results/",                       # saves testing results here.
        ckpt_iter='best'                                                      # which epoch to load to evaluate a model
    )
  
  opt = SimpleNamespace(**opt.__dict__, **opt_extra.__dict__)       # conactenate all options
  opt.phase = 'train' if train else 'test'
  opt.loaded_latest_iter=None;
  
  return opt





def set_dataset_default_lm(opt):
  #set default values based on given dataset
  if opt.dataset_mode == "ade20k":
      opt.lambda_labelmix=10.0
      opt.EMA_decay=0.9999
  if opt.dataset_mode == "cityscapes":
      opt.lr_g=0.0004
      opt.lambda_labelmix=5.0
      opt.freq_fid=2500
      opt.EMA_decay=0.999
  if opt.dataset_mode == "coco":
      opt.lambda_labelmix=10.0
      opt.EMA_decay=0.9999
      opt.num_epochs=100
        
def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    elif opt.which_iter == "best":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "best_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)

def configure_arguments(custom_arguments,train=False):
  #set default values
  opt = get_default_opt(train)
  custom_arguments(opt)
  if train:
    set_dataset_default_lm(opt)
    if opt.continue_train:
      update_options_from_file(opt)
    custom_arguments(opt)
  

  utils.fix_seed(opt.seed)
  print_options(opt, train)
  if train:
    opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
  if train:
    save_options(opt, train)
  return opt


# need to change the other functions
def update_options_from_file(opt):
  new_opt = load_options(opt)
  for k, v in sorted(vars(opt).items()):
    if hasattr(new_opt, k) and v != getattr(new_opt, k):
      new_val = getattr(new_opt, k)
      setattr(opt, k, new_val)


def print_options(opt, train):
    message = ''
    message += '----------------- Options ---------------\n'
    default_opt = get_default_opt(train)
    for k, v in sorted(vars(opt).items()):
        comment = ''
        
        default = getattr(default_opt, k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

def save_options(opt, train):
    path_name = os.path.join(opt.checkpoints_dir,opt.name)
    os.makedirs(path_name, exist_ok=True)
    default_opt = get_default_opt(train)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = getattr(default_opt, k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)
        
        
