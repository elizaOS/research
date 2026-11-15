# TonyPi stock RPC (no custom sidecar)

`@elizaos/plugin-tonypi` talks to the **stock Hiwonder `RPCServer.py`** on the
robot (JSON-RPC HTTP, port **9030**). That is the same transport the bruhOS
mobile app uses.

Do **not** stop `RPCServer.py` / `TonyPi.py` for this plugin — they own the
servo bus. We are a client only.

```bash
export ELIZA_TONYPI_RPC_URL=http://192.168.149.1:9030

# Off the robot AP network, tunnel:
#   ssh -L 9030:127.0.0.1:9030 pi@<robot>
export ELIZA_TONYPI_RPC_URL=http://127.0.0.1:9030
```

## Optional bruhOS enhancement

If bruhOS is running a skill HTTP API beside the robot, set:

```bash
export ELIZA_BRUHOS_URL=http://192.168.149.1:9040
# or: export ELIZA_BRUHOS_AUTO=true   # probes $ELIZA_TONYPI_HOST:9040
```

Absence of bruhOS is normal. Stock TonyPi / TonyPi Pro keep working on RPC alone.

An earlier experimental Python sidecar that opened `/dev/ttyAMA0` itself was
retired — do not revive it while `RPCServer.py` is running.
