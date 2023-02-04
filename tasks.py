"""
Module with invoke tasks
"""

import invoke

import net.invoke.host


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(net.invoke.host)
