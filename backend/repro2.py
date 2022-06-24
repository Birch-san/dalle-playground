from jax.core import Primitive
from jax.interpreters.mlir import _platform_specific_lowerings, _lowerings
import jax.interpreters.xla as xla
from jax.interpreters.xla import _translations
import pickle

platform = 'iree'
primitive: Primitive = pickle.loads(b'\x80\x04\x95Q\x00\x00\x00\x00\x00\x00\x00\x8c\x08jax.core\x94\x8c\rCallPrimitive\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94\x8c\nremat_call\x94\x8c\x04impl\x94h\x00\x8c\tcall_impl\x94\x93\x94ub.')

if primitive in _platform_specific_lowerings[platform]:
  print("oh, seems like it worked; I guess there's been a misunderstanding")
elif primitive in xla._backend_specific_translations[platform]:
  print("oh, seems like it worked; I guess there's been a misunderstanding")
elif primitive in _lowerings:
  print("oh, seems like it worked; I guess there's been a misunderstanding")
elif primitive in xla._translations:
  print("oh, seems like it worked; I guess there's been a misunderstanding")
else:
  raise NotImplementedError(
      f"MLIR translation rule for primitive '{primitive.name}' not "
      f"found for platform {platform}")