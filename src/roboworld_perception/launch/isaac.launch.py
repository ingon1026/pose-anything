"""Isaac Sim 구성 프리셋.

perception.launch.py 는 bag 재생을 기본으로 잡고 있다. Isaac Sim 이 이미지를
직접 발행하는 구성에서는 매번 인자를 여러 개 붙여야 하는데, 빠뜨렸을 때
증상이 조용해서 원인 찾기에 시간이 든다 (마커만 안 보이거나, 며칠 뒤 CUDA
에러로 터지거나). 그래서 실기동으로 검증된 조합을 여기 묶는다.

    ros2 launch roboworld_perception isaac.launch.py

인자는 전부 그대로 덮어쓸 수 있다. 각 값을 왜 이렇게 잡았는지는
docs/isaac_sim_stability_2026-08-14.md 에 실측과 함께 적혀 있다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# (인자, Isaac 기본값, 왜 base 와 다른가)
_PRESET = [
    # Isaac /clock 기준으로 perception과 RViz가 모두 같은 시간축을 쓴다.
    # base launch가 두 Node에 함께 전달하므로 한쪽만 wall time에 남지 않는다.
    ("use_sim_time", "true"),
    # RealSense 드라이버가 없어 camera_link -> camera_color_optical_frame 을
    # 아무도 안 채운다. 끄면 RViz 에 마커가 통째로 안 보인다.
    ("publish_optical_tf", "true"),
    # Isaac 은 RealSense 드라이버가 없어 world->camera_link 도 아무도 안 보낸다.
    # 그러면 rviz 의 Fixed Frame "world" 가 없는 프레임이 되어 아무것도 안 뜬다.
    # publish_optical_tf 를 켠 것과 같은 이유다. bag 재생처럼 TF 트리가 이미
    # 녹화된 경우에는 부모가 둘이 되므로 base 기본값은 false 로 둔다.
    ("publish_world_tf", "true"),
    # 원래 근거는 VRAM 이었다 — 렌더프로덕트 텍스처가 재할당되면 ROS2 발행
    # 노드의 CUDA 핸들이 무효가 되고, 1008 -> 672 로 낮춰 CUDA 에러가
    # 23,410 -> 0 회가 됐다. **그 근거는 이제 성립하지 않는다** (2026-08-24
    # 라이브 실측): 23,410 회는 `rviz=true` 였을 때의 일이고, 현재 프리셋은
    # rviz=false 다. **VRAM 은 병목이 아니다.**
    # ⚠ 여기 적혀 있던 *"1008 피크 **9,797 MiB** / 12GB 중 **2,485 MiB**
    # 남음"* 은 **저장소에 근거가 0건이다**(2026-08-25 확인, grep 0). 출처가
    # 안 갈려서 지우지도 바꾸지도 않고 표시만 한다. 저장소가 가진 1008 값은
    # 두 열이고 인용할 때 어느 열인지 밝혀야 한다 — **8,869 MiB(여유 3,413)**
    # = 17:04(출하 조건에 가까운) 열, **9,585 MiB(여유 2,697)** = 구독 QoS 가
    # 빠졌던 HEAD 열. **셋 중 어느 값이든 결론(병목 아님)은 같다.**
    # 상세는 docs/image_size_2026-08-21.md ③
    #
    # 지금 672 를 쓰는 실제 이유는 **속도**다. 라이브 실측(전제 검사 통과):
    # 672 에서 **4.98 Hz**, 1008 에서 **2.68 Hz (-46.1%)**. Isaac 실시간은
    # 이미 느려서(detect_interval=1 인 이유가 그것) 2.7 Hz 면 추적기가 설
    # 자리를 더 잃는다.
    #
    # ⚠ **[2026-08-25] 라벨 정정 — 위 두 열이 뒤바뀌어 있었다.**
    # 여기 적혀 있던 *"현재 HEAD(2026-08-24 저녁) 4.72 / 2.72 (-42.4%)"* 는
    # **구독 QoS(BEST_EFFORT/depth=1)가 빠져 있던 빌드** 값이라 이제 낡았다 —
    # `0b819de` 로 그 QoS 가 커밋으로 되살아나면서, *"이후 폐기된 빌드"* 로
    # 적혀 있던 **17:04 의 4.98 / 2.68 쪽이 출하 조건에 가깝다.** QoS 가
    # 빠지면서 672 만 5% 손해를 봤던 그 차이다.
    # **이건 재측정이 아니라 라벨 정정이다** — 17:04 빌드는 지금 트리와
    # **바이트 동일하지 않으므로**(그 사이 다른 변경도 있다) *"가장 가까운
    # 빌드"* 로만 읽을 것이고, **정확한 재측정은 하지 않았다.**
    # **672 결정은 -42% 든 -46% 든 바뀌지 않는다** — 어느 열이든 격차가
    # 40%대다. 절대 Hz 를 인용할 때만 어느 빌드인지 밝히면 된다.
    # 상세는 docs/image_size_2026-08-21.md ③ (docs/README.md §2·§4·§5)
    #
    # VRAM 은 판단 근거가 못 된다 — 1008 에서도 2.7GB 가 남는다(이 2,697 MiB
    # 는 낡은 쪽 열의 값이고, 17:04 열은 3,413 MiB 다. 어느 쪽이든 남는다).
    #
    # 새 값을 고를 때는 **patch_size(14)의 배수여야 한다.** 배수가 아니면
    # score 가 골짜기를 만든다 — test2 를 `cell phone` 으로 재면 784=0.81,
    # 800=**0.34**, 812=0.67 이다. 800 은 쓰면 안 된다. 672=14x48,
    # 896=14x64, 1008=14x72 는 전부 배수다.
    #
    # ## 672 의 대가 (2026-08-21 / 08-24 실측)
    # **소물체가 조용히 사라진다.** test2 에서 스마트폰(겉보기 최단축 39px)이
    # 672 에서 통째로 없어진다. 큰 물체(109~117px)는 멀쩡하다 — 소물체만
    # 선택적으로 죽고 에러가 없어서 알아채기 어렵다.
    #
    # 이 씬은 블록만 있어서(겉보기 33px 인데 살아남는다 — 이유 미확인)
    # 성립하지만, **작은 물체를 추가하면 조용히 사라진다. 그때 해상도를
    # 올리기 전에 어휘를 먼저 강화할 것** — 같은 스마트폰이 `smartphone`
    # 으로는 672 에서 검출 0 인데 `cell phone` 으로는 0.77 로 산다.
    # 어휘는 속도 비용이 0 이고, 해상도를 올리는 것은 위의 발행률을 내는
    # 길이다 (672->1008 이 -46.1%. 896 은 아직 안 쟀다).
    #
    # **치수 정답은 씬 USD 에 있다** — 블록 200 x 55 x 55mm, 벨트 위 가시
    # 높이 54.5mm(0.5mm 잠김). 출처는 isaac_sim_connection_2026-08-13 §2 이고
    # 2026-08-13 부터 적혀 있었다.
    #
    # **폭만 해상도에 반응한다** (isaac_belt_moving 500프레임):
    #     672 -> 46.8mm (-8.2)   1008 -> 49.8mm (-5.2)   정답 55.0
    # 저해상도에서 마스크가 줄어서이고 마스크 점 개수도 6% 적다(515 vs 550).
    # 길이는 194.9 / 195.7 로 0.8mm 차이인데 이것도 해상도 효과로 봐야 한다 —
    # **평면 변동은 풋프린트를 못 움직인다.** 풋프린트는 평면 기저 (u,v) 투영이라
    # 법선 δ 에 cos δ 로 2차 반응한다: 0.35° 면 200mm 에 0.004mm 다. 평면 변동이
    # 걸리는 것은 두께(quantile(points @ n + d))뿐이고 거기엔 d 에 1차로 붙는다.
    # 두께는 57.8 / 57.7 로
    # **차이가 0.1mm — 변동 아래라 "같다" 가 아니라 "구분되지 않는다"** 다.
    # 폭 3.0mm 만 변동의 7배 이상이라 유효하다.
    #
    # 그리퍼 허용 오차가 3mm 보다 타이트하면 1008 을 검토할 것. 다만 1008 로
    # 올려도 폭은 -5.2mm 남는다. 상세는 docs/image_size_2026-08-21.md
    #
    # 주의: 해상도별로 치수를 비교할 때 **CSV 컬럼 인덱스로 대응시키면 안
    # 된다.** match_axes() 가 실행마다 re1(길이)/re2(폭) 배정을 바꾼다 —
    # 같은 bag·같은 해상도에서 실행 길이만 달라도 바뀐다. 두께는 평면 법선
    # 방향이라 re3 로 안정적이다. 이걸 놓쳐서 "두께 -18% 과소" 라는 오측이
    # 한 번 문서에 올라갔다.
    ("image_size", "672"),
    # Isaac 실시간 입력은 SAM3 를 함께 돌리면 0.5 Hz 까지 떨어진다. 기본 5 면
    # SAM3 가 10 초에 한 번 도는 셈이라 추적기가 설 틈이 없다.
    # 실측: detect_interval=8 -> 평균 0.1 개, =1 -> 평균 4.7 개.
    ("detect_interval", "1"),
    # 최신 RGB-D 쌍만 SAM3로 넘긴다. 동기화기가 여러 쌍을 보관하면 처리 중
    # 밀린 프레임을 나중에 꺼내 과거 pose를 만들 수 있다.
    ("sync_queue_size", "1"),
    # detect_interval=1 에 RViz 까지 얹으면 12 GB GPU 에서 넘친다.
    # 실측: CUDA 에러 37,151 회 뒤 [omni.rtx] GPU crash, Isaac SIGSEGV.
    # 영상만 볼 거면 grab_debug.py 쪽이 안전하다.
    ("rviz", "false"),
    # base 기본값 1 은 벨트 위 블록을 하나만 잡는다. 씬에는 8 개가 올라가 있어
    # 나머지 7 개가 조용히 버려진다 — 에러가 없어서 알아채기 어렵다.
    # 실측(2026-08-19, 96 프레임): 프레임당 고유 트랙 평균 7.7 개 / 최대 8 개.
    # 8 에 딱 맞추면 블록이 하나 더 들어오거나 마스크가 갈릴 때 잘리므로 여유를 둔다.
    ("max_per_prompt", "10"),
    # `publish_score_min` 은 **프리셋에서 뺐다** (Isaac 은 2026-08-24 까지 0.6).
    # base 기본값이 이미 0.0 이라 여기 다시 적으면 무동작이고, 아래
    # generate_launch_description() 주석이 금지한 패턴이다 — base 가 바뀌면
    # Isaac 만 조용히 옛 값에 남는다.
    #
    # 왜 껐나: 게이트 off CSV 재검증에서 정상 블록 최저 score 가 조각의
    # 최저(0.645)보다 **낮았다** — 정상을 살리면서 조각을 막는 문턱이 존재하지
    # 않는다. ⚠ 여기 적혀 있던 `0.135` 는 **조건이 빠졌다**(2026-08-25 정정):
    # 그것은 `output/belt_moving`(**merge off**)의 **"경계 절단 T8 제외"** 열
    # 값이다. 현행 프리셋은 `enable_merge=true` 라 볼 열은 `belt_merge` 이고,
    # 정상 블록 최저는 **0.112**(실블록 전체) / T8 을 빼면 0.420 이다
    # (docs/datasets.md "게이트 불변식" 표). **결론은 어느 값으로도 성립한다**
    # — 넷 다 조각 최저 0.645 아래다.
    # 0.6 의 실제 손익은 정상 90 프레임 손실 대 조각 0 프레임 차단이고,
    # 조각을 막고 있는 것은 아래 enable_merge 다. 근거와 "다시 켜려면 어떻게
    # 측정할 것인가" 는 docs/datasets.md 의 "게이트 불변식" 절에 있다.

    # 한 물체에 트랙이 둘 붙는 중복을 사후 삭제한다. 생성 시점 가드는
    # "태어날 때 남의 박스 안"만 막아서, 떨어져 태어나 나중에 같은 자리로
    # 수렴한 중복은 통과한다 — 실측(isaac_belt_moving): EndStop_03 에
    # 트랙 2개가 1703프레임 공존하며 생존 트랙의 z-std 를 0.72 -> 3.58mm 로
    # 5배 악화시켰다. 발행만 막아서는 안 되는 이유가 이것이다 — 중복 트랙이
    # 매칭 층에서 계속 검출을 뺏어간다.
    #
    # `enable_merge` 는 **프리셋에서 뺐다.** base 기본값이 이미 true 라
    # (perception.launch.py:42) 여기 다시 적으면 무동작이고, 위
    # publish_score_min 과 같은 이유로 금지된 패턴이다 — base 가 바뀌면
    # Isaac 만 조용히 옛 값에 남는다. 실제로 그랬다: 5d687bf 가 값을
    # 뒤집었는데 여기 주석은 "base 기본값이 false" 로 2026-08-26 까지
    # 남아 있었다.
    #
    # 왜 켜도 안전한지(KAPPA_PHYS 가 벨트 씬 전용이라는 것 포함)와 다른
    # 씬에 이식할 때 무엇을 다시 봐야 하는지는 정본에 있다 —
    # tracker.py:258 의 IouTracker.__init__ enable_merge 주석.
    # 프롬프트 A/B 실측(2026-08-19, 12 종 x 20 초, 재현성 ±0.2 mm)으로 고른 값.
    # 기하는 어휘에 전혀 반응하지 않는다 — 검출된 9 종 모두 우측 3 개의 길이가
    # 152.1/154.9/154.6 mm 로 0.4 mm 이내 동일했다(그 원인은 geometry 의 depth
    # 클립이고 별도로 고쳤다). 따라서 프롬프트로 고를 수 있는 것은 두 가지뿐이다:
    #   조각 트랙의 유무, 그리고 publish_score_min 대비 score 여유.
    #   "blue plastic block"  조각 1~2 개(프레임마다 들락) / score 0.937·0.888
    #   "blue bar with holes" 조각 0 개(60 초 291 프레임 전부) / score 0.929·0.880
    # 조각을 공짜로 없애면서 score 손실이 0.01 이하라 이쪽을 기본값으로 올린다.
    # 탈락: "blue plastic beam" 은 조각 0 이지만 최저 score 0.60 으로 게이트에
    # 닿는다. "blue plastic blocks"(복수형)는 score 가 게이트 아래로 내려가
    # 발행이 통째로 멈춘다 — 복수형은 쓰지 말 것.
    ("prompts", "blue bar with holes"),
]

def generate_launch_description():
    # base 의 나머지 인자는 다시 적지 않는다. IncludeLaunchDescription 은
    # 설정 스코프를 새로 만들지 않고, DeclareLaunchArgument 는 아직 값이
    # 없을 때만 기본값을 넣는다. 그래서 CLI 로 준 값은 그대로 base 까지
    # 흘러가고, 안 준 것은 base 자신의 기본값이 살아난다.
    # 여기서 굳이 다시 적으면 base 의 기본값을 옛 값으로 못 박게 된다 —
    # score_threshold 를 옛 값으로 고정해 버리는 식의 조용한 사고가 난다.
    # (프리셋에 올리는 것은 실측 근거가 있는 값만. 위 _PRESET 참고.)
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=v) for n, v in _PRESET]
        + [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("roboworld_perception"), "launch",
                 "perception.launch.py"])))]
    )
