Optional Lifers tools plugins (disabled by default).

Enable: set stack.json "plugins"."enabled" to true, or export LIFERS_PLUGINS=1.

Layout (under LIFERS_ROOT, default rel_dir tools/plugins):

  tools/plugins/<plugin_name>/plugin.py

Each plugin module must define:

  def register_plugin_tools(registry, root: pathlib.Path) -> None:
      registry.register(MyTool())  # lifers.tools.Tool subclasses

Do not ship demo plugins that register extra tools unless tests use an isolated temp directory.
