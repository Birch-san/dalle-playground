# https://nixos.wiki/wiki/Development_environment_with_nix-shell
# { pkgs ? import <nixpkgs> {} }:
#   pkgs.mkShell {
# }
let
  rust-overlay = (import (builtins.fetchTarball
    "https://github.com/oxalica/rust-overlay/archive/6bc59b9c4ad1cc1089219e935aa727a96d948c5d.tar.gz"));

  # Pinned nixpkgs, deterministic. Last updated: 2022-06-06T17:31:00Z [nixos-unstable].
  pkgs = import (fetchTarball(
    "https://github.com/NixOS/nixpkgs/archive/30d1a2f29e9b309533567dbe55d5ea72653cc6f9.tar.gz")) {
      overlays = [ rust-overlay ];
    };

  # https://github.com/oxalica/rust-overlay#cheat-sheet-common-usage-of-rust-bin
  # rust-nightly = pkgs.rust-bin.selectLatestNightlyWith (toolchain: toolchain.default);
  # Wrap only `cargo-expand`.
  # cargo-expand = pkgs.writeShellScriptBin "cargo-expand" ''
  #   export RUSTC="${rust-nightly}/bin/rustc";
  #   export CARGO="${rust-nightly}/bin/cargo";
  #   exec "${pkgs.cargo-expand}/bin/cargo-expand" "$@"
  # '';
in pkgs.mkShell {
  name = "...";

  packages = with pkgs; [
    # ...
    # (rust-bin.stable.latest.default.override {
    #   extensions = ["rust-src"];
    # })
    rust-bin.beta.latest.minimal
    pkgs.darwin.apple_sdk.frameworks.Security
    # cargo-expand
    # ...
  ];
}