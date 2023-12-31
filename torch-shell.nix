{ bootstrap ? import <nixpkgs> {} }:

let
    # pkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/dfd8f84aef129f1978e446b5d45ef05cd4421821.tar.gz";
    # pkgs_source = ~/repo/nixpkgs;
    #pkgs_source = bootstrap.fetchFromGitHub { # for safety of checking the hash
    #    owner = "stites";
    #    repo = "nixpkgs";
    #    rev = "pytorch_11";
    #    sha256 = "1w1bdqnpjcgdmql3jfcmyz0g0wix4xg17417a7ask5bsphbhpia2";
    #  };
    #pkgs_source = fetchTarball "https://github.com/NixOS/nixpkgs/archive/dfa8e8b9bc4a18bab8f2c55897b6054d31e2c71b.tar.gz";
    pkgs_source = fetchTarball "https://github.com/nixos/nixpkgs/archive/nixos-21.05.tar.gz";
    overlays = [
      (self: super:  # define our local packages
         {
          python3 = super.python3.override {
           packageOverrides = python-self: python-super: {
	            sacremoses = python-self.callPackage /opt/nix/sacremoses-0.0.35.nix {};
	            sentencepiece = python-self.callPackage /opt/nix/sentencepiece-0.1.84.nix {};
#              transformers = python-self.callPackage /opt/nix/transformers-4.0.0.nix { };
              torchtext = python-self.callPackage /opt/nix/torchtext-0.4.0.nix { };
           };};})
      (import /opt/nix/nvidia-410.78.nix)  # fix version of nvidia drivers
      (self: super: {
          cudatoolkit = super.cudatoolkit_10; # fix version of cuda
          cudnn = super.cudnn_cudatoolkit_10;})
    ];
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
    pkgs = import pkgs_source {inherit overlays; inherit config;};
    py = pkgs.python3;
    pyEnv = py.buildEnv.override {
      extraLibs = with py.pkgs;
        [
         # If you want to have a local virtualenv, see here: https://github.com/NixOS/nixpkgs/blob/master/doc/languages-frameworks/python.section.md
         pytorch
	 matplotlib
	 sentencepiece
	 transformers
	 networkx
	 torchtext
	 toolz
         ipython
	 tqdm
	 numpy
        ];
      ignoreCollisions = true;};
in
  pkgs.stdenv.mkDerivation {
    name = "sh-env";
    buildInputs = [pyEnv];
    shellHook = ''
      export LANG=en_US.UTF-8
      export PYTHONIOENCODING=UTF-8
      export LD_PRELOAD=/lib64/libcuda.so.1:/lib64/libnvidia-fatbinaryloader.so.418.74  
     '';
  }
