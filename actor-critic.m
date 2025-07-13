% @author: SH Cho
clear all; close all force; format compact

function [env, nFeatures, nOutputs, success_threshold] = create_env()
  env = py.gymnasium.make('CartPole-v1', pyargs('render_mode', "rgb_array"));
  obv_space = env.observation_space; 
  state_dim =  double(env.observation_space.shape(1));
  action_dim = double(env.action_space.n);  
  success_threshold = env.spec.reward_threshold;
  disp("success_threshold: "+ success_threshold)
  nFeatures = state_dim;
  nOutputs = action_dim;
end

function net = create_network(nFeatures, nHidden, nOutputs)
  net = dlnetwork;
  layers = [ 
    featureInputLayer(nFeatures, 'Name', 'input_layer');
    fullyConnectedLayer(nHidden, 'Name', 'fc1_pi'); 
    reluLayer("Name","pi_relu1");
    fullyConnectedLayer(nOutputs, 'Name', 'fc2_pi');
    softmaxLayer('Name', 'pi_softmax_out'); % pi is for the policy function.
  ];  
  net = addLayers(net,layers);
  layers = [ 
    fullyConnectedLayer(1, 'Name', 'fc1_v'); % v is for the value function.
  ];  
  net = addLayers(net,layers);
  net = connectLayers(net, "pi_relu1", "fc1_v"); 
  layout = networkDataLayout([nFeatures NaN],"CB");
  net = initialize(net, layout);
  layers = net.Layers
  learnables = net.Learnables
  %disp_weights(net.Layers)
  if 0
    connector.ensureServiceOn;
    analyzeNetwork(net)
  end
end

function [gradients, loss] = modelGradients(net, buffer, nFeatures, discount_factor)
  rewards = buffer.Reward(:);
  actions = buffer.Action(:);
  dlX = single(buffer.State);
  batch_size = size(buffer.State, 3);
  dlX = dlarray(reshape(dlX, nFeatures, batch_size), 'CB');
  [pi_s, values] = forward(net, dlX, 'Outputs', ["pi_softmax_out", "fc1_v"]);
  log_probs = []; 
  for i = 1:size(pi_s, 2)
    temp = pi_s(:, i); 
    temp2 = temp(actions(i));
    log_prob = log(temp2);
    log_probs = [log_probs, log_prob];
  end 
  discounted_rewards = []; 
  R = 0;
  for i = length(rewards):-1:1
    r = rewards(i);
    R = r + discount_factor * R;
    discounted_rewards = [R, discounted_rewards];
  end
  std_discounted_rewards = std(discounted_rewards);
  if std_discounted_rewards == 0
    std_discoundte_rewards = eps;
  end
  discounted_rewards = (discounted_rewards - mean(discounted_rewards)) / std_discounted_rewards;
  actor_loss = 0;
  for i = 1:length(log_probs)
      temp = -log_probs(i) * discounted_rewards(i);
      actor_loss = actor_loss + temp;
  end
  critic_loss = 0;
  for i = 1:length(values)
      temp = huber(values(i), discounted_rewards(i), 'TransitionPoint', 1, 'DataFormat', 'CB');
      critic_loss = critic_loss + temp;
  end
  loss = actor_loss + critic_loss;
  gradients = dlgradient(loss, net.Learnables);
  %disp_gradients(gradients)
end

function disp_gradients(gradients)
 for l = 1:length(gradients.Value)
  disp("  " + gradients.Layer(l) + " gradients -> " + gradients.Parameter(l) + ": " + mat2str(gradients.Value{l, :}, 4))
 end
end

function disp_weights(layers)
 for l = 1:length(layers)
  if isprop(layers(l), 'Weights') 
   disp([layers(l).Name, ' -> Weight: ', mat2str(layers(l).Weights, 8)])
  end
  if isprop(layers(l), 'Bias')
   disp([layers(l).Name, ' -> Bias: ', mat2str(layers(l).Bias, 8)])
  end
 end
end

function action = categorical(p)
    p_cumsum = [];
    for i = 1:length(p)
      p_cumsum(i) = sum(p(1:i));
    end
    temp_uniform = rand;
    temp2 = p_cumsum >= temp_uniform;;
    [val, action] = max(temp2);
end

pe = pyenv();
[env, nFeatures, nOutputs, success_threshold] = create_env();
nHidden = 256
SEED = 1
rng(SEED);
net = create_network(nFeatures, nHidden, nOutputs);

discount_factor = 0.99;
learningRate = 0.001;
averageGrad = [];
averageSqGrad = [];
buffer.State = {};
buffer.Action = {};
buffer.Reward = [];
buffer.NextState = {};
buffer.IsDone = [];

max_num_episodes = 2000;
max_step = 1000;
print_interval = 10;
scores_array = [];
scores_array_max_length = 100;

for episode = 1:max_num_episodes
    obv = env.reset(pyargs('seed', int8(SEED)));
    state = double(obv{1});
    step = 0;
    score = 0;
    while step < max_step
        step = step+1;
        dlX = dlarray(single(state'), 'CB');
        pi_s = forward(net, dlX, 'Outputs', ["pi_softmax_out"]);
        action = categorical(pi_s);
        action_python = int8(action - 1);
        obs = env.step(action_python);
        next_state = double(obs{1});
        reward = double(obs{2});
        done = int8(obs{3});
        score = score + reward;
        if done == true
            reward = -1;
        end
        if length(buffer.State) == 0
            buffer.State = state';
            buffer.Action = action;
            buffer.Reward = [buffer.Reward, reward];
            buffer.NextState = next_state';
            buffer.IsDone = [buffer.IsDone, done];
        else
            buffer.State = cat(3, buffer.State, state');
            buffer.Action = cat(3, buffer.Action, action);
            buffer.Reward = [buffer.Reward, reward];
            buffer.NextState = cat(3, buffer.NextState, next_state');
            buffer.IsDone = [buffer.IsDone, done];
        end
        if done == true
            break;
        end
        state = next_state;
    end
    scores_array = [scores_array, score];
    if length(scores_array) > scores_array_max_length
        scores_array(1) = [];
    end
    avg_score = mean(scores_array);
    if episode == 1 || rem(episode, print_interval) == 0
        disp("Episode " + episode + " step " + step + ", last " + length(scores_array) + " avg score = " + round(avg_score,4))
    end
    if avg_score > success_threshold
        disp("Solved after " + episode + " episodes. Average Score: "  + round(avg_score,4))
        break
    end
    [gradients, loss] = dlfeval(@modelGradients, net, buffer, nFeatures, discount_factor);
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, averageGrad, averageSqGrad, episode, learningRate);
    buffer.State = {};
    buffer.Action = {};
    buffer.Reward = [];
    buffer.NextState = {};
    buffer.IsDone = [];
end
env.close()
