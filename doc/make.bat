@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build
set I18NSPHINXOPTS=%SPHINXOPTS% %SOURCEDIR%
set I18NSPHINXLANGS=-l zh_CN

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html_zh_cn
%SPHINXBUILD% -b html %ALLSPHINXOPTS% -t zh_cn -D language='zh_CN' %SOURCEDIR% %BUILDDIR%/html_zh_cn
:gettext
%SPHINXBUILD% -b gettext %I18NSPHINXOPTS% %BUILDDIR%/locale
if errorlevel 1 exit /b 1
%SPHINXINTL% update -p %BUILDDIR%/locale %I18NSPHINXLANGS%
if errorlevel 1 exit /b 1
python %SOURCEDIR%/norm_zh.py
if errorlevel 1 exit /b 1
echo.
echo.Build finished. The message catalogs are in %BUILDDIR%/locale.
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
