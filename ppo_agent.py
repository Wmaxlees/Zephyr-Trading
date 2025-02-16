import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

def get_crypto_data():
    """
    Placeholder function to fetch daily OHLCV data for top N cryptocurrencies.
    You'll need to integrate with a crypto data API (e.g., CoinGecko, Binance API).
    Returns a dictionary or Pandas DataFrame with OHLCV data for each coin.
    """
    data = {}
    coins = ['btc']
    for coin in coins:
        coin_data = pd.read_csv(f"normalized_data/{coin}.csv")
        coin_data.fillna(0, inplace=True)
        data[coin] = coin_data
    return data


def split_data(data_dict, train_ratio=0.8):
    """
    Splits a dictionary of DataFrames into training and validation sets.

    Args:
        data_dict (dict): A dictionary where keys are coin names and values are Pandas DataFrames.
        train_ratio (float): The ratio of data to use for training (0.0 to 1.0).

    Returns:
        tuple: A tuple containing two dictionaries: (train_data_dict, val_data_dict).
               Each dictionary has the same keys as the input, but the values are
               the split DataFrames for training and validation, respectively.
    """
    train_data_dict = {}
    val_data_dict = {}

    # Ensure all dataframes have the same length
    first_df_len = len(list(data_dict.values())[0])
    for df in list(data_dict.values()):
        if len(df) != first_df_len:
            raise ValueError("All DataFrames in the dictionary must have the same length.")
        

    num_samples = first_df_len
    train_size = int(num_samples * train_ratio)

    for coin, df in data_dict.items():
        train_data_dict[coin] = df[:train_size]
        val_data_dict[coin] = df[train_size:]

    return train_data_dict, val_data_dict


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embedding_dim):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Identity() # No token embeddings in this case, input is already numerical features
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim
        )
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        length = tf.shape(inputs)[1] # Get sequence length from input dynamically
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions # Add position embeddings to input


class CryptoTradingEnvironment:
    def __init__(self, data, initial_capital=1000, n_days=5):
        self.data = data # Preprocessed data
        self.coins = list(data.keys())
        self.initial_capital = initial_capital
        self.current_step = 0
        self.portfolio_value_history = []
        self.n_days = n_days

    def reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.holdings = {coin: 0 for coin in self.coins} # Units held for each coin
        self.portfolio_value_history = [self.capital]
        return self._get_state()

    def _get_state(self):
        """
        Returns the current state representation (features) as a TensorFlow tensor.
        Extract features for the current timestep from self.data and portfolio state.
        """
        current_state = [self.capital]
        historical_data = []
        for coin, df in self.data.items():
            price, coin_state = self._get_coin_state(df, self.current_step, self.n_days)
            current_state.append(price.numpy().item())
            historical_data.append(coin_state)
        
        # Concatenate states into a single tensor
        historical_data = tf.concat(historical_data, axis=1)

        current_state = tf.constant(current_state, dtype=tf.float32)
        current_state = tf.concat([list(self.holdings.values()), current_state], axis=0)

        return current_state, tf.transpose(historical_data, [1, 0])
    
    def _get_coin_state(self, data, t, n_days):
        """
        Gets the state representation from the BTC data.

        Args:
            data (pd.DataFrame): DataFrame containing BTC data.
            t (int): Current time step.
            n_days (int): Number of days to consider for the state.

        Returns:
            tensor: State representation.
        """
        d = t - n_days + 1
        block = data.iloc[d: t + 1] if d >= 0 else pd.concat([data.iloc[0:1]] * (-d) + [data.iloc[0: t + 1]])  # Pad with initial data if d < 0

        block = block[[
            'open_norm',
            'high_norm',
            'low_norm',
            'close_norm',
            'volume_norm',
            'ema_12_norm',
            'ema_26_norm',
            'macd_norm',
            'macd_signal_norm',
            'macd_hist_norm',
            'bb_upper_norm',
            'bb_middle_norm',
            'bb_lower_norm',
            'rsi_norm',
            'sma_20_norm',
            'sma_50_norm',
            'atr_norm',
            'obv_norm']]

        # Convert block to numpy array
        block = block.to_numpy()

        prev_coin_data = []

        for i in range(1, len(block[0])):
            prev_coin_data.append(tf.constant([block[:,i]], dtype=tf.float32))

        current_price = tf.constant(block[-1][3], dtype=tf.float32)

        return current_price, tf.concat(prev_coin_data, axis=0)

    def step(self, actions):
        """
        Executes a step using a *dictionary* of actions.

        Args:
            actions (dict):  A dictionary where keys are 'cash' and coin names,
                             and values are the desired portfolio weights (0.0 to 1.0).

        Returns:
             tuple: (next_state, reward, done, info)
        """

        # 0. Input Validation and Action Normalization (CRITICAL)
        if 'cash' not in actions:
            raise ValueError("Actions dictionary must include 'cash' key.")
        for coin in self.coins:
            if coin not in actions:
                raise ValueError(f"Actions dictionary must include '{coin}' key.")

        action_values = [actions['cash']] + [actions[coin] for coin in self.coins]
        total_action_value = sum(action_values)
        if not np.isclose(total_action_value, 1.0):
            # Normalize actions to ensure they sum to 1.0
            normalized_actions = [val / total_action_value for val in action_values]
            actions['cash'] = normalized_actions[0]
            for i, coin in enumerate(self.coins):
                actions[coin] = normalized_actions[i + 1]
        else:
             normalized_actions = action_values


        # 1. Calculate current portfolio value (before rebalancing)
        current_portfolio_value = self.capital
        coin_prices = []
        for coin, df in self.data.items():
            price = df.iloc[self.current_step]['close']
            coin_prices.append(price)
            current_portfolio_value += self.holdings[coin] * price


        # 2. Rebalance portfolio (CRITICAL CHANGES)
        self.capital = actions['cash'] * current_portfolio_value  # Update cash

        for i, coin in enumerate(self.coins):
            target_value_in_coin = actions[coin] * current_portfolio_value
            self.holdings[coin] = target_value_in_coin / coin_prices[i]  # Buy/sell to target *shares*

        # Update portfolio weights *after* rebalancing
        self.portfolio_weights = np.array(normalized_actions)


        # 3. Advance to the next time step
        self.current_step += 1
        next_state = self._get_state()


        # 4. Calculate the *new* portfolio value (after price changes)
        next_portfolio_value = self.capital
        for i, coin in enumerate(self.coins):
            next_price = self.data[coin].iloc[self.current_step]['close']
            next_portfolio_value += self.holdings[coin] * next_price

        # 5. Calculate reward (log return)
        reward = 0.0
        if next_portfolio_value > 0:
            reward = np.log(current_portfolio_value / next_portfolio_value)
        # reward = np.maximum(0, next_portfolio_value - current_portfolio_value)
        # reward = (current_portfolio_value - next_portfolio_value) / current_portfolio_value

        # 6. Check if episode is done
        done = self.current_step >= len(list(self.data.values())[0]) - 1 or next_portfolio_value == 0
        info = {}

        self.portfolio_value_history.append(next_portfolio_value)

        return next_state, reward, done, info

    def render(self):
        """
        Optional: Visualize portfolio value history or other relevant information.
        """
        # --- Implement visualization if desired ---
        pass


# --- 3. PPO Agent (TensorFlow/Keras Example - Adapt for PyTorch if preferred) ---
class PPOAgent:
    def __init__(self, state_dim, hist_dim, action_dim):
        self.state_dim = state_dim
        self.hist_dim = hist_dim
        self.action_dim = action_dim
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-3) # Tune LR
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-3) # Tune LR
        self.clip_param = 0.2 # PPO clip parameter - Tune

    def _build_actor_model(self):
        """
        Builds the actor neural network with a Transformer Encoder layer and Positional Embeddings.
        """
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)
        time_steps = self.hist_dim[0]
        num_features = self.hist_dim[1]

        # --- Positional Embedding Layer ---
        positional_embedding_layer = PositionalEmbedding(
            sequence_length=time_steps, embedding_dim=num_features # Embedding dim matches feature dimension
        )
        x = positional_embedding_layer(hist_input_layer) # Apply positional embeddings

        # --- Transformer Encoder Block ---
        # Parameters for the Transformer Encoder
        head_size = num_features  # Number of features, can be adjusted
        num_heads = 2   # Number of attention heads
        ff_dim = 64     # Hidden layer size in feed forward network

        # Layer Normalization and Multi-Head Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1  # Added dropout for regularization
        )(x, x)  # Self-attention
        x = tf.keras.layers.Add()([hist_input_layer, attn_output]) # Residual connection (note: using hist_input_layer here is intentional after positional embedding)

        # Feed Forward Network
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output) # Project back to the input feature dimension
        transformer_output = tf.keras.layers.Add()([x, ffn_output]) # Residual connection

        # --- Flatten and Dense Layers for Value Output ---
        flattened = tf.keras.layers.Flatten()(transformer_output) # Flatten the output of the transformer
        concatenated = tf.keras.layers.Concatenate()([flattened, state_input_layer]) # Concatenate with current state
        dense3 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        dense4 = tf.keras.layers.Dense(64, activation='tanh')(dense3)

        mean_output = tf.keras.layers.Dense(self.action_dim, activation='softmax', name="mean_output")(dense4)
        stddev_output = tf.keras.layers.Dense(self.action_dim, activation=lambda x: tf.nn.softplus(x) + 1e-8, name="stddev_output")(dense4)

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=[mean_output, stddev_output])

    def _build_critic_model(self):
        """
        Builds the critic neural network with a Transformer Encoder layer and Positional Embeddings.
        """
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)
        time_steps = self.hist_dim[0]
        num_features = self.hist_dim[1]

        # --- Positional Embedding Layer ---
        positional_embedding_layer = PositionalEmbedding(
            sequence_length=time_steps, embedding_dim=num_features # Embedding dim matches feature dimension
        )
        x = positional_embedding_layer(hist_input_layer) # Apply positional embeddings

        # --- Transformer Encoder Block ---
        # Parameters for the Transformer Encoder
        head_size = num_features  # Number of features, can be adjusted
        num_heads = 2   # Number of attention heads
        ff_dim = 64     # Hidden layer size in feed forward network

        # Layer Normalization and Multi-Head Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1  # Added dropout for regularization
        )(x, x)  # Self-attention
        x = tf.keras.layers.Add()([hist_input_layer, attn_output]) # Residual connection (note: using hist_input_layer here is intentional after positional embedding)

        # Feed Forward Network
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output) # Project back to the input feature dimension
        transformer_output = tf.keras.layers.Add()([x, ffn_output]) # Residual connection

        # --- Flatten and Dense Layers for Value Output ---
        flattened = tf.keras.layers.Flatten()(transformer_output) # Flatten the output of the transformer
        concatenated = tf.keras.layers.Concatenate()([flattened, state_input_layer]) # Concatenate with current state
        dense3 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        dense4 = tf.keras.layers.Dense(64, activation='tanh')(dense3)

        value_layer = tf.keras.layers.Dense(1, activation='tanh')(dense4)

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=value_layer)

    def _positional_encoding(self, seq_length, d_model):
        """
        Generates the positional encoding as described in "Attention is All You Need".

        Args:
            seq_length: The length of the sequence.
            d_model: The dimensionality of the encoding (should match the embedding dimension).

        Returns:
            A tensor of shape (1, seq_length, d_model) containing the positional encoding.
        """
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pe = tf.zeros((seq_length, d_model), dtype=tf.float32)

        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)

        # Interleave sin and cos
        pe_sin = tf.reshape(pe_sin, (seq_length, d_model // 2))
        pe_cos = tf.reshape(pe_cos, (seq_length, d_model // 2))
        pe = tf.concat([pe_sin, pe_cos], axis=-1)

        return tf.expand_dims(pe, 0)  # Add a batch dimension


    def get_action(self, state):
        """
        Samples action from the actor network based on the current state.
        Returns action, log probability of the action.
        """
        current_state = tf.expand_dims(state[0], axis=0)
        hist_state = tf.expand_dims(state[1], axis=0)
        mean, stddev = self.actor_model((current_state, hist_state,))
        # Sample action from normal distribution parameterized by mean and stddev
        normal_dist = tfp.distributions.Normal(mean, stddev) # Requires TensorFlow Probability
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action)
        return action.numpy()[0], log_prob.numpy()[0]

    def train_step(self, states, actions, advantages, log_probs_old, returns):
        """
        Performs one training step of PPO algorithm: updates actor and critic networks.
        """
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        advantages_tensor_expanded = tf.expand_dims(advantages_tensor, axis=1) # Expand to (64, 1)
        advantages_tensor_expanded = tf.tile(advantages_tensor_expanded, [1, self.action_dim]) # Tile to (64, 2) - assuming action_dim=2
        log_probs_old_tensor = tf.convert_to_tensor(log_probs_old, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        # --- Critic Training ---
        with tf.GradientTape() as tape_critic:
            values_predicted = self.critic_model(states)
            critic_loss = tf.keras.losses.MeanSquaredError()(returns_tensor, values_predicted) # MSE loss
        grads_critic = tape_critic.gradient(critic_loss, self.critic_model.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic_model.trainable_variables))

        # --- Actor Training ---
        with tf.GradientTape() as tape_actor:
            mean, stddev = self.actor_model(states)
            normal_dist = tfp.distributions.Normal(mean, stddev)
            log_probs_new = normal_dist.log_prob(actions)
            # print(log_probs_new)
            ratio = tf.exp(log_probs_new - log_probs_old_tensor)
            policy_loss_unclipped = -ratio * advantages_tensor_expanded # Use expanded advantages
            policy_loss_clipped = -tf.clip_by_value(ratio, 1-self.clip_param, 1+self.clip_param) * advantages_tensor_expanded # Use expanded advantages
            actor_loss = tf.reduce_mean(tf.maximum(policy_loss_unclipped, policy_loss_clipped)) # PPO Clipped loss
        grads_actor = tape_actor.gradient(actor_loss, self.actor_model.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor_model.trainable_variables))

        return actor_loss, critic_loss


# --- 4. Training Loop ---
if __name__ == '__main__':
    # --- Hyperparameters (Tune these!) ---
    episodes = 1000 # Number of training episodes
    timesteps_per_episode = 200 # Length of each episode
    batch_size = 64 # Batch size for training updates
    gamma = 0.99 # Discount factor
    gae_lambda = 0.95 # GAE lambda parameter

    # --- Get Data ---
    crypto_data = get_crypto_data()
    train_data, val_data = split_data(crypto_data)
    test_data = val_data

    # --- Environment and Agent Setup ---
    env = CryptoTradingEnvironment(train_data, n_days=14) # Use training data for environment
    current_state_dim = (3,)
    hist_state_dim = (14, 17,)
    action_dim = len(env.coins) + 1
    agent = PPOAgent(current_state_dim, hist_state_dim, action_dim)

    # --- Training ---
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = []
        episode_current_states, episode_hist_states, episode_actions, episode_log_probs, episode_rewards_gae, episode_values = [], [], [], [], [], []

        for timestep in range(timesteps_per_episode):
            action, log_prob = agent.get_action(state)

            action_set = {}
            action_set['cash'] = action[0]
            action_set['btc'] = action[1]

            next_state, reward, done, _ = env.step(action_set)

            episode_current_states.append(state[0])
            episode_hist_states.append(state[1])
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)

            current_state, hist_state = state  # Unpack the tuple
            current_state = np.expand_dims(current_state.numpy(), axis=0)
            hist_state = np.expand_dims(hist_state.numpy(), axis=0)

            value_prediction = agent.critic_model([current_state, hist_state])
            episode_values.append(value_prediction[0,0].numpy()) # Predict value for GAE

            state = next_state
            if done:
                break

        # --- Calculate GAE (Generalized Advantage Estimation) ---
        current_state, hist_state = next_state
        current_state = np.expand_dims(current_state.numpy(), axis=0)
        hist_state = np.expand_dims(hist_state.numpy(), axis=0)
        values_next = agent.critic_model((current_state, hist_state))[0,0].numpy() if not done else 0
        episode_values.append(values_next)
        advantages = []
        gae = 0.0
        for t in reversed(range(len(episode_rewards))): # Reverse iterate for GAE calculation
            delta = episode_rewards[t] + gamma * episode_values[t+1] - episode_values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae) # Insert scalar gae directly (REVISED LINE - NO REPEAT)
        episode_rewards_gae = np.array(advantages)

        # --- Prepare batch data ---
        current_states_batch = np.array(episode_current_states)
        hist_states_batch = np.array(episode_hist_states)
        actions_batch = np.array(episode_actions)
        log_probs_batch = np.array(episode_log_probs)
        returns_batch = episode_rewards_gae + np.array(episode_values[:-1]) # Target returns = Advantages + Value baseline

        # --- Train agent in batches (optional - can train after each episode or mini-batches) ---
        num_batches = len(episode_current_states) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_current_states = current_states_batch[start_idx:end_idx]
            batch_hist_states = hist_states_batch[start_idx:end_idx]
            batch_actions = actions_batch[start_idx:end_idx]
            batch_advantages = episode_rewards_gae[start_idx:end_idx]
            batch_log_probs = log_probs_batch[start_idx:end_idx]
            batch_returns = returns_batch[start_idx:end_idx]

            actor_loss, critic_loss = agent.train_step((batch_current_states, batch_hist_states,), batch_actions, batch_advantages, batch_log_probs, batch_returns)
            print(f"Episode {episode+1}, Batch {batch_idx+1}/{num_batches}, Actor Loss: {actor_loss.numpy():.4f}, Critic Loss: {critic_loss.numpy():.4f}")


        avg_reward = np.mean(episode_rewards)
        print(f"Episode {episode+1}/{episodes}, Average Reward: {avg_reward:.4f}")

    # --- 5. Evaluation (Placeholder - Implement evaluation on test_data) ---
    # --- Evaluate trained agent on test_data and calculate performance metrics ---
    print("Training finished. Evaluate agent on test data...")
    # --- Implement evaluation loop and metrics calculation here ---