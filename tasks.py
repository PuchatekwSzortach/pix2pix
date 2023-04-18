"""
Module with invoke tasks
"""

import invoke

import net.invoke.host
import net.invoke.visualize
import net.invoke.tests
import net.invoke.train


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(net.invoke.host)
ns.add_collection(net.invoke.visualize)
ns.add_collection(net.invoke.tests)
ns.add_collection(net.invoke.train)
