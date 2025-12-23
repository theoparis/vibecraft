{
  inputs = {
    # rust 1.92: https://github.com/NixOS/nixpkgs/pull/470993
    nixpkgs.url = "github:nixos/nixpkgs";
  };

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://cache.garnix.io"
    ];

    extra-trusted-public-keys = [
      "cache.nixos.org-1:CNHJZBh9K4tP3EKF6FkkgeVYsS3ohTl+oS0Qa8bezVs="
      "cache.garnix.io:CTFPyKSLcx5RMJKfLo5EEPUObbA78b0YQ2DTCJXqr9g="
    ];
  };

  outputs = {
    self,
    nixpkgs,
    ...
  }: let
    eachSystem = nixpkgs.lib.genAttrs [
      "x86_64-linux"
      "aarch64-linux"
      "riscv64-linux"
    ];
  in {
    devShells = eachSystem (
      system:
        with nixpkgs.legacyPackages.${system}; {
          default =
            (mkShell.override {
              stdenv = useWildLinker clangStdenv;
            })
            rec {
              nativeBuildInputs = [
                pyrefly
                buck2
                # reindeer
                rustup
                pkg-config
                alejandra
                nil
                python3
                wild
              ];

              buildInputs = [
                wayland
                alsa-lib
                libxkbcommon
                vulkan-loader
              ];

              LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
            };
        }
    );
  };
}
