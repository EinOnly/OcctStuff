# 安装 tcl-tk（如果还没装）
brew install tcl-tk pyenv

# 设置编译环境变量
export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/tcl-tk/lib/pkgconfig"

# 安装支持 tkinter 的 Python 3.10
env PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I/opt/homebrew/opt/tcl-tk/include' --with-tcltk-libs='-L/opt/homebrew/opt/tcl-tk/lib -ltcl8.6 -ltk8.6'" \
pyenv install 3.10.13

# 使用这个版本作为默认
pyenv global 3.10.13

# 验证 tkinter 是否可用
python -m tkinter