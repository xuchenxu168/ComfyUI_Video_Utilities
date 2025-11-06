"""
Video Utilities Server Routes
提供视频转码和预览服务
"""

import os
import re
import asyncio
import subprocess
import shutil

try:
    import folder_paths
    from aiohttp import web
    from server import PromptServer

    print("✅ Video Utilities: Server modules imported successfully")

    # 获取 ffmpeg 路径
    ffmpeg_path = None
    try:
        from videohelpersuite.utils import ffmpeg_path as vhs_ffmpeg_path
        ffmpeg_path = vhs_ffmpeg_path
        print(f"✅ Video Utilities: Using VHS ffmpeg: {ffmpeg_path}")
    except:
        # 尝试从系统路径获取
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"✅ Video Utilities: Using system ffmpeg: {ffmpeg_path}")

    if ffmpeg_path is None:
        print("⚠️ Video Utilities: FFmpeg not found. Video preview transcoding will be disabled.")

    ENCODE_ARGS = {'encoding': 'utf-8', 'errors': 'ignore'}

    def is_safe_path(path, strict=False):
        """检查路径是否安全"""
        if not path or not os.path.exists(path):
            return False
        
        try:
            input_dir = folder_paths.get_input_directory()
            output_dir = folder_paths.get_output_directory()
            temp_dir = folder_paths.get_temp_directory()
            
            real_path = os.path.realpath(path)
            
            allowed_dirs = [
                os.path.realpath(input_dir),
                os.path.realpath(output_dir),
                os.path.realpath(temp_dir)
            ]
            
            for allowed_dir in allowed_dirs:
                if real_path.startswith(allowed_dir):
                    return True
            
            return False
        except Exception as e:
            print(f"❌ Error checking path safety: {e}")
            return False

    async def resolve_video_path(query):
        """从查询参数中解析视频路径"""
        if 'filename' not in query:
            return web.Response(status=400, text="Missing filename parameter")

        filename = query['filename']
        file_type = query.get('type', 'output')
        subfolder = query.get('subfolder', '')

        print(f"🔍 Resolving video path:")
        print(f"   - filename: {filename}")
        print(f"   - type: {file_type}")
        print(f"   - subfolder: {subfolder}")

        # 确定基础目录
        if file_type == 'input':
            base_dir = folder_paths.get_input_directory()
        elif file_type == 'temp':
            base_dir = folder_paths.get_temp_directory()
        elif file_type == 'upload':
            base_dir = folder_paths.get_input_directory()
        else:  # output
            base_dir = folder_paths.get_output_directory()

        print(f"   - base_dir: {base_dir}")

        # 构建完整路径
        if subfolder:
            file_path = os.path.join(base_dir, subfolder, filename)
        else:
            file_path = os.path.join(base_dir, filename)

        print(f"   - file_path: {file_path}")
        print(f"   - exists: {os.path.exists(file_path)}")

        # 安全检查
        if not is_safe_path(file_path):
            print(f"❌ Access denied: path not safe")
            return web.Response(status=403, text=f"Access denied: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return web.Response(status=404, text=f"File not found: {file_path}")

        print(f"✅ Path resolved successfully")
        return file_path, filename, base_dir

    @PromptServer.instance.routes.get("/video_utilities/viewvideo")
    async def view_video_transcoded(request):
        """
        视频预览端点 - 实时转码为 WebM 格式
        支持 MPEG-4 等浏览器不兼容的格式
        """
        query = request.rel_url.query

        print(f"🎬 Video Utilities: Received request for video: {query.get('filename', 'unknown')}")
        print(f"🎬 Full query params: {dict(query)}")

        # 解析视频路径
        try:
            path_res = await resolve_video_path(query)
            if isinstance(path_res, web.Response):
                print(f"❌ Video Utilities: Path resolution failed")
                print(f"❌ Response status: {path_res.status}")
                print(f"❌ Response text: {path_res.text}")
                return path_res
        except Exception as e:
            print(f"❌ Exception in resolve_video_path: {e}")
            import traceback
            traceback.print_exc()
            return web.Response(status=500, text=f"Error: {str(e)}")
        
        file_path, filename, output_dir = path_res
        print(f"✅ Video Utilities: Resolved path: {file_path}")
        
        # 如果没有 ffmpeg，直接返回文件
        if ffmpeg_path is None:
            print("⚠️ Video Utilities: FFmpeg not available, returning file directly")
            if is_safe_path(output_dir, strict=True):
                return web.FileResponse(path=file_path)
            else:
                return web.Response(status=500, text="FFmpeg not available")
        
        # 检测视频编码
        try:
            probe_cmd = [
                ffmpeg_path, "-v", "quiet", "-i", file_path,
                "-t", "0", "-f", "null", "-"
            ]
            proc = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )
            _, stderr = await proc.communicate()
            
            stderr_text = stderr.decode(**ENCODE_ARGS)
            
            codec_match = re.search(r': Video: (\w+)', stderr_text)
            fps_match = re.search(r', (\d+(?:\.\d+)?) fps,', stderr_text)
            
            codec_name = codec_match.group(1) if codec_match else 'unknown'
            base_fps = float(fps_match.group(1)) if fps_match else 30
            
            print(f"🎬 Video Utilities: Transcoding {filename} (codec: {codec_name}, fps: {base_fps})")
            
        except Exception as e:
            print(f"❌ Error probing video: {e}")
            base_fps = 30
            codec_name = 'unknown'
        
        # 构建 FFmpeg 转码命令 - 使用 H.264 编码输出 MP4 格式（更兼容）
        # 参考 VHS (VideoHelperSuite) 的转码方式
        args = [
            ffmpeg_path,
            "-v", "error",
            "-i", file_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保宽高是偶数（H.264要求）
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",  # 降低CRF以提高质量（23是推荐值）
            "-pix_fmt", "yuv420p",  # 确保像素格式兼容
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",  # 音频采样率
            "-ac", "2",  # 双声道
            "-movflags", "frag_keyframe+empty_moov+faststart",  # 优化流式传输
            "-f", "mp4",
            "-"
        ]
        
        try:
            print(f"🎬 Video Utilities: Starting FFmpeg transcode...")
            print(f"🎬 FFmpeg command: {' '.join(args)}")

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL
            )

            print(f"✅ FFmpeg process started (PID: {proc.pid})")

            # 创建一个任务来读取 stderr
            async def log_stderr():
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    print(f"[FFmpeg stderr] {line.decode(**ENCODE_ARGS).strip()}")

            stderr_task = asyncio.create_task(log_stderr())

            try:
                resp = web.StreamResponse()
                resp.content_type = 'video/mp4'
                resp.headers["Content-Disposition"] = f'inline; filename="{os.path.splitext(filename)[0]}.mp4"'
                resp.headers["Accept-Ranges"] = "bytes"
                resp.headers["Cache-Control"] = "no-cache"
                resp.headers["Connection"] = "keep-alive"
                await resp.prepare(request)

                chunk_count = 0
                total_bytes = 0
                while True:
                    chunk = await proc.stdout.read(2**20)
                    if not chunk:
                        break
                    await resp.write(chunk)
                    chunk_count += 1
                    total_bytes += len(chunk)
                    if chunk_count == 1:
                        print(f"✅ First chunk sent ({len(chunk)} bytes)")

                await proc.wait()
                await stderr_task

                print(f"✅ Video Utilities: Transcode completed")
                print(f"   - Chunks sent: {chunk_count}")
                print(f"   - Total bytes: {total_bytes}")
                print(f"   - FFmpeg exit code: {proc.returncode}")

                if proc.returncode != 0:
                    print(f"❌ FFmpeg exited with error code: {proc.returncode}")

            except (ConnectionResetError, ConnectionError, BrokenPipeError) as e:
                proc.kill()
                stderr_task.cancel()
                print(f"⚠️ Client disconnected during video streaming: {e}")

            return resp

        except Exception as e:
            print(f"❌ Error transcoding video: {e}")
            import traceback
            traceback.print_exc()
            return web.Response(status=500, text=f"Transcoding error: {str(e)}")

    @PromptServer.instance.routes.get("/video_utilities/test")
    async def test_endpoint(request):
        """测试端点是否工作"""
        return web.json_response({
            'status': 'ok',
            'message': 'Video Utilities server is working!',
            'ffmpeg_path': ffmpeg_path
        })

    @PromptServer.instance.routes.get("/video_utilities/detect_codec")
    async def detect_codec(request):
        """检测视频编码格式"""
        try:
            filename = request.rel_url.query.get("filename", "")
            file_type = request.rel_url.query.get("type", "input").lower()
            subfolder = request.rel_url.query.get("subfolder", "")

            if not filename:
                return web.json_response({
                    'error': 'No filename provided',
                    'needs_transcode': False
                }, status=400)

            # 确定文件路径
            if file_type == "output":
                base_dir = folder_paths.get_output_directory()
            else:
                base_dir = folder_paths.get_input_directory()

            # 如果有 subfolder，添加到路径中
            if subfolder:
                file_path = os.path.join(base_dir, subfolder, filename)
            else:
                file_path = os.path.join(base_dir, filename)

            print(f"🔍 detect_codec API 调用:")
            print(f"   - filename: {filename}")
            print(f"   - file_type: {file_type}")
            print(f"   - subfolder: {subfolder}")
            print(f"   - base_dir: {base_dir}")
            print(f"   - file_path: {file_path}")
            print(f"   - exists: {os.path.exists(file_path)}")

            # 安全检查
            if not is_safe_path(file_path):
                print(f"❌ detect_codec: 路径不安全")
                return web.json_response({
                    'error': 'Invalid file path',
                    'needs_transcode': False
                }, status=403)

            if not os.path.exists(file_path):
                print(f"❌ detect_codec: 文件不存在")
                return web.json_response({
                    'error': 'File not found',
                    'needs_transcode': False
                }, status=404)

            # 使用 ffprobe 检测编码
            if not ffmpeg_path:
                return web.json_response({
                    'error': 'FFmpeg not available',
                    'needs_transcode': False,
                    'codec': 'unknown'
                })

            ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
            probe_cmd = [
                ffprobe_path, "-v", "quiet", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name", "-of", "csv=p=0", file_path
            ]

            print(f"🔍 detect_codec: 执行 ffprobe 命令: {' '.join(probe_cmd)}")
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
            print(f"🔍 detect_codec: ffprobe 返回码: {result.returncode}")
            print(f"🔍 detect_codec: ffprobe 输出: {result.stdout.strip()}")
            print(f"🔍 detect_codec: ffprobe 错误: {result.stderr.strip()}")

            if result.returncode == 0:
                codec_name = result.stdout.strip().lower()

                # 判断是否需要转码
                # MPEG-4 part 2 (mpeg4) 需要转码，因为浏览器支持有限
                needs_transcode = codec_name in ['mpeg4', 'msmpeg4v3', 'msmpeg4v2', 'msmpeg4']

                print(f"✅ detect_codec: 编码={codec_name}, 需要转码={needs_transcode}")

                return web.json_response({
                    'codec': codec_name,
                    'needs_transcode': needs_transcode,
                    'filename': filename
                })
            else:
                return web.json_response({
                    'error': 'Failed to detect codec',
                    'needs_transcode': False,
                    'codec': 'unknown'
                })

        except Exception as e:
            print(f"❌ Codec detection error: {e}")
            import traceback
            traceback.print_exc()
            return web.json_response({
                'error': str(e),
                'needs_transcode': False,
                'codec': 'unknown'
            }, status=500)

    # 注释掉拦截 /api/view 的代码，因为我们已经有了 /video_utilities/viewvideo 端点
    # 并且访问 routes._resources 会导致 AttributeError
    # original_view_handler = None
    # for route in PromptServer.instance.routes._resources:
    #     if hasattr(route, '_path') and route._path == '/view':
    #         for route_info in route:
    #             if route_info.method == 'GET':
    #                 original_view_handler = route_info.handler
    #                 break
    #         break

    # @PromptServer.instance.routes.get("/api/view")
    # async def intercept_view(request):
    #     """拦截 /api/view 请求，视频文件转发到转码端点"""
    #     query = request.rel_url.query
    #     filename = query.get('filename', '')
    #
    #     # 检查是否是视频文件
    #     video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    #     is_video = any(filename.lower().endswith(ext) for ext in video_extensions)
    #
    #     if is_video:
    #         print(f"🎬 Intercepting /api/view for video: {filename}")
    #         print(f"🎬 Redirecting to /video_utilities/viewvideo")
    #         # 转发到转码端点
    #         return await view_video_transcoded(request)
    #     else:
    #         # 非视频文件，使用原始处理器
    #         if original_view_handler:
    #             return await original_view_handler(request)
    #         else:
    #             return web.Response(status=404, text="Not found")

    print("✅ Video Utilities server routes loaded successfully")
    print("✅ Test endpoint: http://127.0.0.1:8188/video_utilities/test")
    # print("✅ Intercepting /api/view for video files")  # 已禁用拦截功能

except Exception as e:
    print(f"❌ Video Utilities: Failed to load server routes: {e}")
    import traceback
    traceback.print_exc()

