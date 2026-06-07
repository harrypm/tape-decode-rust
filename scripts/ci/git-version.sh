#!/bin/sh
set -eu

V="${DECODE_LIGHT_VERSION_OVERRIDE:-${DECODE_LIGHT_VERSION:-}}"

if [ -z "$V" ]; then
	V=$(git describe --tags --dirty --match 'v*' --match 'decode-light-*' 2>/dev/null || true)
fi

if [ -z "$V" ]; then
	SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
	DIRTY=""
	if git diff --quiet --ignore-submodules -- 2>/dev/null; then
		:
	else
		DIRTY="-dirty"
	fi
	V="dev-${SHA}${DIRTY}"
fi

case "$V" in
	decode-light-*)
		V="${V#decode-light-}"
		;;
esac

printf '%s\n' "$V"
