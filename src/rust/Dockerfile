ARG ARCH=aarch64
ARG SDK_VERSION=12.9.0
ARG UBUNTU_VERSION=24.04

FROM axisecp/acap-native-sdk:${SDK_VERSION}-${ARCH}-ubuntu${UBUNTU_VERSION} AS builder

# Install host build tools (needed for Rust proc-macro / build script compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ pkg-config libclang-dev clang \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Add aarch64 cross-compilation target
RUN rustup target add aarch64-unknown-linux-gnu

WORKDIR /opt/app

# Detect sysroot path and create cross-compilation config
RUN . /opt/axis/acapsdk/environment-setup* && \
    SYSROOT="${SDKTARGETSYSROOT}" && \
    mkdir -p .cargo && \
    printf '[target.aarch64-unknown-linux-gnu]\n\
linker = "aarch64-linux-gnu-gcc"\n\
rustflags = [\n\
    "-C", "link-arg=--sysroot=%s",\n\
    "-C", "link-arg=-L%s/usr/lib",\n\
]\n\
\n\
[env]\n\
PKG_CONFIG_PATH = "%s/usr/lib/pkgconfig:%s/usr/share/pkgconfig"\n\
PKG_CONFIG_SYSROOT_DIR = "%s"\n\
PKG_CONFIG_ALLOW_CROSS = "1"\n' \
    "$SYSROOT" "$SYSROOT" "$SYSROOT" "$SYSROOT" "$SYSROOT" \
    > .cargo/config.toml

# Set pkg-config env vars so -sys crate build scripts find Axis SDK libs
# These must be ENV (not just .cargo/config.toml [env]) for build.rs scripts
ENV PKG_CONFIG_PATH="/opt/axis/acapsdk/sysroots/aarch64/usr/lib/pkgconfig:/opt/axis/acapsdk/sysroots/aarch64/usr/share/pkgconfig"
ENV PKG_CONFIG_SYSROOT_DIR="/opt/axis/acapsdk/sysroots/aarch64"
ENV PKG_CONFIG_ALLOW_CROSS="1"

# Copy all source at once (no cache warming — build.rs must run with real source)
COPY Cargo.toml .
COPY src/ src/
COPY manifest.json .
COPY LICENSE .

# Copy model
RUN mkdir -p model
COPY model/ model/

# Build the real binary — pkg-config will find liblarod + vdostream and link them
RUN cargo build --release --target aarch64-unknown-linux-gnu 2>&1

# Copy binary to packaging location
RUN cp target/aarch64-unknown-linux-gnu/release/autoacap .

# Create a no-op Makefile so acap-build doesn't fail on the make step
RUN echo 'all:' > Makefile && echo '	@true' >> Makefile

# Package as .eap (acap-build needs the SDK environment)
RUN . /opt/axis/acapsdk/environment-setup* && \
    acap-build . -m manifest.json -a model
