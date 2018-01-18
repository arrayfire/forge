if (-Not (Test-Path -Path C:\projects\dependencies\glbinding\lib\glbinding.lib -PathType leaf))
{
  if (-Not (Test-Path -Path C:\projects\glbinding-dev.zip -PathType leaf))
  {
    (new-object net.webclient).DownloadFile('https://github.com/cginternals/glbinding/releases/download/v2.1.1/glbinding-2.1.1-msvc2015-x64-dev.zip', 'c:/projects/glbinding-dev.zip')
  }
  Expand-Archive -LiteralPath C:\projects\glbinding-dev.zip -DestinationPath C:\projects\dependencies\glbinding
}

if (-Not (Test-Path -Path C:\projects\dependencies\glbinding\glbinding.dll -PathType leaf))
{
  if (-Not (Test-Path -Path C:\projects\glbinding-rt.zip -PathType leaf))
  {
    (new-object net.webclient).DownloadFile('https://github.com/cginternals/glbinding/releases/download/v2.1.1/glbinding-2.1.1-msvc2015-x64-runtime.zip', 'c:/projects/glbinding-rt.zip')
  }
  Expand-Archive -LiteralPath C:\projects\glbinding-rt.zip -DestinationPath C:\projects\dependencies\glbinding
}

if (-Not (Test-Path -Path C:\projects\dependencies\glm\build\package\include\glm\glm.hpp -PathType leaf))
{
  if (-Not (Test-Path -Path C:\projects\glm.zip -PathType leaf))
  {
    (new-object net.webclient).DownloadFile('https://github.com/g-truc/glm/releases/download/0.9.8.5/glm-0.9.8.5.zip', 'c:/projects/glm.zip')
  }
  Expand-Archive -LiteralPath C:\projects\glm.zip -DestinationPath C:\projects\dependencies
}

if (-Not (Test-Path -Path C:\projects\dependencies\glfw-3.2.1\build\package\lib\glfw3.dll -PathType leaf))
{
  if (-Not (Test-Path -Path C:\projects\glfw3.zip -PathType leaf))
  {
    (new-object net.webclient).DownloadFile('https://github.com/glfw/glfw/releases/download/3.2.1/glfw-3.2.1.zip', 'c:/projects/glfw3.zip')
  }
  Expand-Archive -LiteralPath C:\projects\glfw3.zip -DestinationPath C:\projects\dependencies
}

if (-Not (Test-Path -Path C:\projects\dependencies\FreeImage\Dist\x64\FreeImage.dll -PathType leaf))
{
  if (-Not (Test-Path -Path C:\projects\freeimage.zip -PathType leaf))
  {
    (new-object net.webclient).DownloadFile('http://downloads.sourceforge.net/freeimage/FreeImage3170Win32Win64.zip', 'c:/projects/freeimage.zip')
  }
  Expand-Archive -LiteralPath C:\projects\freeimage.zip -DestinationPath C:\projects\dependencies
}