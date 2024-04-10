import re
import sys
from fruit_nerf.scripts.exporter import entrypoint
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(entrypoint())
