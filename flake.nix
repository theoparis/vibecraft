{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
    ];

    extra-trusted-public-keys = [
      "cache.nixos.org-1:CNHJZBh9K4tP3EKF6FkkgeVYsS3ohTl+oS0Qa8bezVs="
    ];
  };

  outputs = {
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
