import numpy as np
import matplotlib.pyplot as plt

# 환경 설정
world_size = 100.0
landmarks = np.array([[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]])
N_landmarks = len(landmarks)

# 로봇 클래스
class Robot:
    def __init__(self):
        self.x = np.random.rand() * world_size
        self.y = np.random.rand() * world_size
        self.orientation = np.random.rand() * 2.0 * np.pi
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 5.0
    
    def set(self, new_x, new_y, new_orientation):
        self.x = new_x
        self.y = new_y
        self.orientation = new_orientation
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = new_f_noise
        self.turn_noise = new_t_noise
        self.sense_noise = new_s_noise
    
    def sense(self):
        Z = []
        for i in range(N_landmarks):
            dist = np.sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += np.random.normal(0.0, self.sense_noise)
            Z.append(dist)
        return Z
    
    def move(self, turn, forward):
        orientation = self.orientation + float(turn) + np.random.normal(0.0, self.turn_noise)
        orientation %= 2 * np.pi
        dist = float(forward) + np.random.normal(0.0, self.forward_noise)
        x = self.x + np.cos(orientation) * dist
        y = self.y + np.sin(orientation) * dist
        x %= world_size  # 경계 조건 처리
        y %= world_size  # 경계 조건 처리
        result = Robot()
        result.set(x, y, orientation)
        result.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return result

# 파티클 클래스
class Particle(Robot):
    def __init__(self, x=None, y=None, orientation=None):
        super().__init__()
        if x is not None and y is not None and orientation is not None:
            self.set(x, y, orientation)
        self.weight = 1.0

    def move(self, turn, forward):
        # 부모 클래스의 move 메소드를 호출하여 로봇의 위치를 업데이트
        result = super().move(turn, forward)
        # Particle 인스턴스를 생성하여 반환
        particle = Particle(result.x, result.y, result.orientation)
        particle.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        particle.weight = self.weight  # 현재 파티클의 가중치를 유지
        return particle
    
    def measurement_prob(self, measurement):
        # 센서 측정값과 파티클이 예측한 값 사이의 확률을 계산
        prob = 1.0
        for i in range(N_landmarks):
            dist = np.sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob
    
    def Gaussian(self, mu, sigma, x):
        # 가우시안 확률 밀도 함수
        return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))


def particle_filter(motions, N=500):
    # 파티클들 초기화
    particles = [Particle(np.random.rand() * world_size, np.random.rand() * world_size, np.random.rand() * 2.0 * np.pi) for i in range(N)]
    
    
    for motion in motions:
        # 모든 파티클을 이동시키기
        particles = [p.move(motion[0], motion[1]) for p in particles]
        
        # 센서 측정값 얻기 (로봇의 실제 이동 후 위치에서 측정)
        Z = myrobot.sense()
        
        # 각 파티클의 가중치 업데이트
        weights = [p.measurement_prob(Z) for p in particles]
        
        # 리샘플링: 가중치에 비례하여 파티클을 선택
        indices = list(range(N))
        resampled_particles = []
        index = int(np.random.rand() * N)
        beta = 0.0
        mw = max(weights)
        for i in range(N):
            beta += np.random.rand() * 2.0 * mw
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % N
            resampled_particles.append(particles[index])
        particles = resampled_particles
    
    # 최종 파티클 집합 반환
    return particles

# 메인 실행 부분
if __name__ == '__main__':
    # 로봇 초기화 및 노이즈 설정
    myrobot = Robot()
    myrobot.set_noise(0.05, 0.05, 5.0)

    # 이동 명령어 설정 ([회전각, 전진거리])
    motions = [[0.1, 5.0] for _ in range(20)]

    # 파티클 필터 실행
    particles = particle_filter(motions, N=500)

    # 시각화를 위한 코드
    plt.figure(figsize=(10, 10))
    plt.xlim(0, world_size)
    plt.ylim(0, world_size)
    for i in range(N_landmarks):
        plt.plot(landmarks[i][0], landmarks[i][1], 'ro')  # 랜드마크 위치 표시
    for particle in particles:
        plt.plot(particle.x, particle.y, 'bo', alpha=0.5)  # 파티클 위치 표시
    plt.show()
