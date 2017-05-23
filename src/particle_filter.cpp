/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	default_random_engine gen;

	// Create normal distributions for x, y and psi.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	weights.resize(num_particles);
	particles.resize(num_particles);

	for (int i = 0; i < num_particles; ++i) {
		weights[i] = 1.0;
		particles[i] = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	default_random_engine gen;

	// Create normal distributions for x, y and psi.
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		if (yaw_rate == 0) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			double next_theta = particles[i].theta + yaw_rate * delta_t;
			double v_yr = velocity / yaw_rate;
			particles[i].x += v_yr * (sin(next_theta) - sin(particles[i].theta));
			particles[i].y += v_yr * (-cos(next_theta) + cos(particles[i].theta));
			particles[i].theta = next_theta;
		}
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i=0; i < observations.size(); ++i) {
		double min_dist = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
		int min_id = 0;
		for (int j = 1; j < predicted.size(); ++j) {
			double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				min_id = j;
			}
		}
		observations[i].id = min_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];
	double denom = 2 * M_PI * sigma_x * sigma_y;

	for (int i = 0; i < num_particles; ++i) {

		// get landmarks in range
		vector<LandmarkObs> predicted;
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			double dist2lm = dist(particles[i].x, particles[i].y,
								  map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			if (dist2lm < sensor_range) {
				LandmarkObs lo = {map_landmarks.landmark_list[j].id_i,
								  map_landmarks.landmark_list[j].x_f,
								  map_landmarks.landmark_list[j].y_f};
				predicted.push_back(lo);
			}
		}

		// skip particle if can't see any landmark
		if (predicted.size() == 0) {
			particles[i].weight = 0;
			weights[i] = 0;
			break;
		}

		//transform observations from local to global frame
		vector<LandmarkObs> trans_obs;
		trans_obs.resize(observations.size());
		for (int j = 0; j < observations.size(); ++j) {
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;
			trans_obs[j].x = obs_x * cos(particles[i].theta) - obs_y * sin(particles[i].theta) + particles[i].x;
			trans_obs[j].y = obs_x * sin(particles[i].theta) + obs_y * cos(particles[i].theta) + particles[i].y;
		}

		// Association
		dataAssociation(predicted, trans_obs);

		// Update particle's weight
		double final_weight = 1;
		for (int j = 0; j < trans_obs.size(); ++j) {
			int lm_id = trans_obs[j].id;
			double diff_x = trans_obs[j].x - predicted[lm_id].x;
			double diff_y = trans_obs[j].y - predicted[lm_id].y;
			double weight = exp(-0.5 * (pow(diff_x / sigma_x, 2) + pow(diff_y / sigma_y, 2))) / denom;
			final_weight *= weight;
		}
		particles[i].weight = final_weight;
		weights[i] = final_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;

	discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> new_particles;
	new_particles.resize(num_particles);
	for(int i = 0; i < num_particles; ++i) {
		new_particles[i] = particles[d(gen)];
	}
	particles = new_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
