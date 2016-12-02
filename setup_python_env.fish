set --local script_dir (dirname (realpath (status -f)))
set --export PYTHONPATH "$PYTHONPATH:$script_dir/lib"
