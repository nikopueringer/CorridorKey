"""Add this directory with nuke.pluginAddPath; no installation is performed."""

import nuke

if nuke.GUI:
    nuke.menu("Nodes").addCommand("CorridorKey/CK2 Local or SSH", "import ck2_nuke.ui; ck2_nuke.ui.create_node()")
