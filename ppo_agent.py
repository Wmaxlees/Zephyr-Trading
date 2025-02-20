import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
import math

def get_crypto_data():
    """
    Placeholder function to fetch daily OHLCV data for top N cryptocurrencies.
    You'll need to integrate with a crypto data API (e.g., CoinGecko, Binance API).
    Returns a dictionary or Pandas DataFrame with OHLCV data for each coin.
    """
    data = {}
    coins = ['btc', 'eth', 'xrp', 'sol']
    for coin in coins:
        coin_data = pd.read_csv(f"normalized_data/{coin}.csv")
        coin_data.fillna(0, inplace=True)
        coin_data.drop_duplicates(subset='timestamp', inplace=True)
        data[coin] = coin_data

    common_timestamps = data[coins[0]]['timestamp']

    # Iterate through the rest of the coins
    for coin in coins[1:]:
        common_timestamps = common_timestamps[common_timestamps.isin(data[coin]['timestamp'])]

    # Filter each DataFrame
    for coin in coins:
        data[coin] = data[coin][data[coin]['timestamp'].isin(common_timestamps)]

    print(data['btc'].shape)

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
    def __init__(self, data, initial_capital=1000, n_periods=5, sharpe_window=30, slippage_factor=0.005, stop_loss_pct=0.1, random_start=False): # Added sharpe_window, slippage_factor, and stop_loss_pct
        self.data = data # Preprocessed data
        self.coins = list(data.keys())
        self.initial_capital = initial_capital
        self.current_step = 0
        self.portfolio_value_history = []
        self.n_periods = n_periods # Number of previous periods to consider for state
        self.sharpe_window = sharpe_window # Window for Sharpe Ratio calculation
        self.portfolio_returns = [] # Store portfolio returns for Sharpe Ratio calculation
        self.slippage_factor = slippage_factor # Slippage factor for transaction cost simulation
        self.stop_loss_pct = stop_loss_pct # Stop loss percentage drop from peak
        self.bitcoin_value_history = []  # History of portfolio value if only holding Bitcoin
        self.bitcoin_holdings = 0  # Units of Bitcoin held in baseline strategy
        self.btc_coin_name = 'btc' # Assuming 'btc' is always in coins, for baseline comparison
        self.peak_prices = {coin: None for coin in self.coins} # Track peak prices for stop loss
        self.random_start = random_start # Randomize starting point

    def reset(self):
        self.current_step = 0
        if self.random_start:
            self.current_step = random.randint(0, 5000)
        self.capital = self.initial_capital
        self.holdings = {coin: 0 for coin in self.coins} # Units held for each coin
        self.portfolio_value_history = [self.capital]
        self.portfolio_returns = [] # Reset portfolio returns history at each reset
        self.bitcoin_value_history = [self.initial_capital] # Reset BTC baseline history
        self.peak_prices = {coin: None for coin in self.coins} # Reset peak prices

        # Initialize Bitcoin baseline strategy
        btc_price_initial = self.data[self.btc_coin_name].iloc[self.current_step]['close']
        self.bitcoin_holdings = self.initial_capital / btc_price_initial if btc_price_initial > 0 else 0 # avoid division by zero


        return self._get_state()

    def _get_state(self):
        """
        Returns the current state representation (features) as a TensorFlow tensor.
        Extract features for the current timestep from self.data and portfolio state.
        """
        current_state = []
        historical_data = []
        holdings_in_usd = [self.capital]
        for coin, df in self.data.items():
            price, coin_state = self._get_coin_state(df, self.current_step, self.n_periods)
            current_state.append(price.numpy().item())
            holdings_in_usd.append(self.holdings[coin] * df.iloc[self.current_step]['close'])
            historical_data.append(coin_state)

        holdings_pct = holdings_in_usd
        if np.sum(holdings_in_usd) > 0:
            holdings_pct = holdings_in_usd / np.sum(holdings_in_usd)
        current_state.extend(holdings_pct)

        # Concatenate states into a single tensor
        historical_data = tf.concat(historical_data, axis=0)

        current_state = tf.constant(current_state, dtype=tf.float32)

        return current_state, tf.transpose(historical_data, [1, 0])

    def _get_coin_state(self, data, t, n_periods):
        """
        Gets the state representation from the BTC data.

        Args:
            data (pd.DataFrame): DataFrame containing BTC data.
            t (int): Current time step.
            n_days (int): Number of days to consider for the state.

        Returns:
            tensor: State representation.
        """
        d = t - n_periods + 1
        block = data.iloc[d: t + 1] if d >= 0 else pd.concat([data.iloc[0:1]] * (-d) + [data.iloc[0: t + 1]])  # Pad with initial data if d < 0

        block = block[[
            'open_log_return',
            'high_log_return',
            'low_log_return',
            'close_log_return',
            'volume_log',
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
        ... (rest of docstring) ...
        """

        # 0. Input Validation and Action Normalization (CRITICAL)
        if 'cash' not in actions:
            raise ValueError("Actions dictionary must include 'cash' key.")
        for coin in self.coins:
            if coin not in actions:
                raise ValueError(f"Actions dictionary must include '{coin}' key.")

        action_values = [actions['cash']] + [actions[coin] for coin in self.coins]
        total_action_value = sum(action_values)

        if np.isclose(total_action_value, 0.0):  # Check if total_action_value is close to zero
            normalized_actions = [0.0] * len(action_values) # Set all weights to zero if sum is zero
            actions['cash'] = 1.0
            for i, coin in enumerate(self.coins):
                actions[coin] = 0.0
        elif not np.isclose(total_action_value, 1.0):
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

        # 2. Rebalance portfolio
        self.capital = actions['cash'] * current_portfolio_value  # Update cash

        for i, coin in enumerate(self.coins):
            target_value_in_coin = actions[coin] * current_portfolio_value
            current_price = coin_prices[i]
            # Simulate slippage - Randomly adjust the execution price
            slippage = random.uniform(-self.slippage_factor, self.slippage_factor)
            execution_price = current_price * (1 + slippage)
            self.holdings[coin] = target_value_in_coin / execution_price  # Buy/sell to target *shares*
            self.peak_prices[coin] = execution_price # Reset peak price after rebalance/buy

        # Update portfolio weights *after* rebalancing
        self.portfolio_weights = np.array(normalized_actions)

        # 3. Advance to the next time step
        self.current_step += 1
        next_state = self._get_state()

        # 4. Calculate the *new* portfolio value (after price changes) and check stop-loss
        next_portfolio_value = 0.0
        for i, coin in enumerate(self.coins):
            next_price = self.data[coin].iloc[self.current_step]['close']
            # --- Stop-Loss Logic ---
            if self.holdings[coin] > 0 and self.peak_prices[coin] is not None: # Only check stop loss if we hold the coin and have recorded a peak price
                if next_price < self.peak_prices[coin] * (1 - (action_set[f'{coin}_sl'] * .1)):
                    self.capital += self.holdings[coin] * next_price # Liquidate coin at current price
                    self.holdings[coin] = 0
                    self.peak_prices[coin] = None # Reset peak price after stop-loss
                else:
                    self.peak_prices[coin] = max(self.peak_prices[coin], next_price) # Update peak price if current price is higher
            next_portfolio_value += self.holdings[coin] * next_price
        next_portfolio_value += self.capital

        # 5. Calculate reward (Sortino Ratio)
        reward = 0.0
        if next_portfolio_value > 0 and current_portfolio_value > 0:
            # Scale the reward by a factor to increase learning stability
            portfolio_return = ((next_portfolio_value - current_portfolio_value) / current_portfolio_value)
            self.portfolio_returns.append(portfolio_return)

            if len(self.portfolio_returns) >= self.sharpe_window: # Changed window name
                returns_window = self.portfolio_returns[-self.sharpe_window:] # Changed window name
                avg_return = np.mean(returns_window)
                returns_window = np.array(returns_window)
                negative_returns = returns_window[returns_window < 0] # Filter negative returns
                downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0 # Downside deviation
                sortino_ratio = avg_return / downside_deviation if downside_deviation != 0 else 0 # Sortino Ratio

                reward = sortino_ratio
            else:
                reward = portfolio_return # Use simple return for initial steps before Sortino window is full


        # 6. Check if episode is done
        done = self.current_step >= len(list(self.data.values())[0]) - 1 or next_portfolio_value == 0
        info = {}

        self.portfolio_value_history.append(next_portfolio_value)

        # --- Bitcoin Baseline Tracking ---
        current_btc_price = self.data[self.btc_coin_name].iloc[self.current_step]['close']
        bitcoin_portfolio_value = self.bitcoin_holdings * current_btc_price
        self.bitcoin_value_history.append(bitcoin_portfolio_value)


        return next_state, reward, done, info

    def render(self):
        """
        Optional: Visualize portfolio value history or other relevant information.
        """
        # --- Implement visualization if desired ---
        pass

    def get_current_portfolio_value(self):
        current_portfolio_value = self.capital
        for coin, df in self.data.items():
            price = df.iloc[self.current_step]['close']
            current_portfolio_value += self.holdings[coin] * price

        return current_portfolio_value


# --- 3. PPO Agent ---
class PPOAgent:
    def __init__(self, state_dim, hist_dim, action_dim):
        self.state_dim = state_dim
        self.hist_dim = hist_dim
        self.action_dim = action_dim

        # Actor Network (Policy Network) - Outputs mean and std_dev for each action dimension
        self.actor_model = self._build_actor_model()
        # Critic Network (Value Network) - Outputs state value
        self.critic_model = self._build_critic_model()

        # Optimizers
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=1e-4)

        # PPO Hyperparameters
        self.gamma = 0.99
        self.clip_ratio = 0.2          # PPO clip ratio
        self.vf_coef = 0.5             # Value function coefficient in loss
        self.entropy_coef = 0.01       # Entropy coefficient in loss (for exploration)
        self.max_grad_norm = 0.5       # Gradient clipping
        self.train_epochs = 10          # Number of PPO epochs per training step

    def _build_actor_model(self):
        """Builds the actor model (policy network) for PPO."""
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)

        hist_norm = tf.keras.layers.BatchNormalization()(hist_input_layer)
        state_norm = tf.keras.layers.BatchNormalization()(state_input_layer)

        time_steps = self.hist_dim[0]
        num_features = 32

        cnn1 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(hist_norm)
        cnn2 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn1)
        cnn3 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn2)

        # Positional Embedding Layer
        positional_embedding_layer = PositionalEmbedding(
            sequence_length=time_steps, embedding_dim=num_features
        )
        x = positional_embedding_layer(cnn3)

        # Transformer Encoder Block
        head_size = num_features
        num_heads = 4
        ff_dim = 64

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Add()([cnn3, attn_output])

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        transformer_output = tf.keras.layers.Add()([x, ffn_output])

        # Flatten and Dense Layers
        flattened = tf.keras.layers.Flatten()(transformer_output)
        dense0 = tf.keras.layers.Dense(512, activation='relu')(flattened)
        concatenated = tf.keras.layers.Concatenate()([dense0, state_norm])
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)

        # Output layers for mean and std_dev of actions (independent Gaussian distributions)
        split_pcts_mean = tf.keras.layers.Dense(math.floor(self.action_dim / 2) + 1, activation='softmax')(dense2) # Mean for split percentages (softmax)
        stop_losses_mean = tf.keras.layers.Dense(math.floor(self.action_dim / 2), activation='sigmoid')(dense2) # Mean for stop losses (sigmoid)
        action_means = tf.keras.layers.Concatenate()([split_pcts_mean, stop_losses_mean]) # Concatenate means

        split_pcts_log_std = tf.keras.layers.Dense(math.floor(self.action_dim / 2) + 1, activation='linear')(dense2) # Log std_dev for split percentages
        stop_losses_log_std = tf.keras.layers.Dense(math.floor(self.action_dim / 2), activation='linear')(dense2) # Log std_dev for stop losses
        action_log_stds = tf.keras.layers.Concatenate()([split_pcts_log_std, stop_losses_log_std]) # Concatenate log std_devs

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=[action_means, action_log_stds]) # Output both mean and log_std


    def _build_critic_model(self):
        """Builds the critic model (value network) for PPO."""
        hist_input_layer = tf.keras.layers.Input(shape=self.hist_dim)
        state_input_layer = tf.keras.layers.Input(shape=self.state_dim)

        hist_norm = tf.keras.layers.BatchNormalization()(hist_input_layer)
        state_norm = tf.keras.layers.BatchNormalization()(state_input_layer)

        time_steps = self.hist_dim[0]
        num_features = 32

        cnn1 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(hist_norm)
        cnn2 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn1)
        cnn3 = tf.keras.layers.Conv1D(filters=num_features, kernel_size=12, strides=3, activation='relu')(cnn2)

        # Positional Embedding Layer
        positional_embedding_layer = PositionalEmbedding(
            sequence_length=time_steps, embedding_dim=num_features
        )
        x = positional_embedding_layer(cnn3)

        # Transformer Encoder Block (smaller for critic)
        head_size = num_features
        num_heads = 4
        ff_dim = 32

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1
        )(x, x)
        x = tf.keras.layers.Add()([cnn3, attn_output])

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        transformer_output = tf.keras.layers.Add()([x, ffn_output])

        # Flatten and Concatenate with Action Input
        flattened = tf.keras.layers.Flatten()(transformer_output)
        dense0 = tf.keras.layers.Dense(512, activation='relu')(flattened)
        concatenated = tf.keras.layers.Concatenate()([dense0, state_norm]) # Critic only takes state
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concatenated)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        value_output = tf.keras.layers.Dense(1, activation='linear')(dense2) # State value output

        return tf.keras.Model(inputs=[state_input_layer, hist_input_layer], outputs=value_output)

    def get_action(self, state, noise=True, old_policy=False): # Added old_policy flag
        """Samples action from actor (policy) network - stochastic policy."""
        current_state = tf.expand_dims(state[0], axis=0)
        hist_state = tf.expand_dims(state[1], axis=0)

        action_means, action_log_stds = self.actor_model((current_state, hist_state))
        action_stddevs = tf.exp(action_log_stds) # Convert log_std to std_dev

        # Create normal distributions for each action dimension
        dist = tfp.distributions.Normal(loc=action_means, scale=action_stddevs)

        if old_policy: # If we need action and log_prob from the *old* policy (for PPO ratio calculation during training)
            return dist.mean().numpy()[0], dist.log_prob(dist.mean()).numpy()[0] # Return mean (deterministic action) and log_prob at mean
        else:
            # Sample action from distribution (stochastic action for exploration/evaluation)
            sampled_actions = dist.sample()
            clipped_actions = tf.clip_by_value(sampled_actions, 0.0, 1.0) # Clip actions to [0, 1]
            return clipped_actions.numpy()[0] # Return sampled action


    def critic_value(self, state):
        """Evaluates state value using critic network."""
        current_state = tf.expand_dims(state[0], axis=0)
        hist_state = tf.expand_dims(state[1], axis=0)
        value = self.critic_model((current_state, hist_state))
        return value

    def train_step(self, replay_buffer, batch_size):
        """Performs one training step for PPO actor and critic networks."""

        # Get data from replay buffer
        state_batch, hist_state_batch, action_batch, old_prob_batch, advantage_batch, value_target_batch = replay_buffer.ppo_sample(batch_size)

        # --- Train Actor (Policy Network) ---
        with tf.GradientTape() as tape_actor:
            action_means, action_log_stds = self.actor_model((state_batch, hist_state_batch))
            action_stddevs = tf.exp(action_log_stds)
            dist = tfp.distributions.Normal(loc=action_means, scale=action_stddevs)
            new_log_probs = dist.log_prob(action_batch) # Log probabilities of actions *under current policy*

            ratio = tf.exp(new_log_probs - old_prob_batch) # Importance ratio
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            surrogate_obj = tf.minimum(ratio * advantage_batch, clipped_ratio * advantage_batch) # PPO Clipped surrogate objective
            actor_loss = -tf.reduce_mean(surrogate_obj) # Maximize surrogate objective (minimize negative)

            # Entropy bonus for exploration - optional but often helpful in PPO
            entropy = tf.reduce_mean(dist.entropy())
            actor_loss = actor_loss - self.entropy_coef * entropy


        actor_grads = tape_actor.gradient(actor_loss, self.actor_model.trainable_variables)
        actor_grads_clipped, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm) # Gradient clipping
        self.optimizer_actor.apply_gradients(zip(actor_grads_clipped, self.actor_model.trainable_variables))

        # --- Train Critic (Value Network) ---
        with tf.GradientTape() as tape_critic:
            value_predictions = self.critic_model((state_batch, hist_state_batch))
            critic_loss = tf.keras.losses.MeanSquaredError()(value_target_batch, value_predictions) # MSE loss for critic
            critic_loss = critic_loss * self.vf_coef # Value function coefficient

        critic_grads = tape_critic.gradient(critic_loss, self.critic_model.trainable_variables)
        critic_grads_clipped, _ = tf.clip_by_global_norm(critic_grads, self.max_grad_norm) # Gradient clipping
        self.optimizer_critic.apply_gradients(zip(critic_grads_clipped, self.critic_model.trainable_variables))

        return actor_loss, critic_loss


# --- Replay Buffer for PPO ---
class ReplayBuffer:
    def __init__(self, buffer_capacity=100000):
        self.buffer_capacity = buffer_capacity
        self.buffer_counter = 0
        self.state_buffer = None
        self.hist_state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.next_hist_state_buffer = None
        self.done_buffer = None
        self.old_prob_buffer = None # Store log probabilities of actions under old policy
        self.value_buffer = None # Store predicted state values

        self.ep_start_index = 0 # Track start index of current episode

    def record(self, state, hist_state, action, reward, next_state, next_hist_state, done, old_prob, value): # Added old_prob, value
        """Records experience to buffer, initializes buffer on first record."""
        if self.buffer_counter == 0:
            self.state_buffer = np.zeros((self.buffer_capacity, state[0].shape[0]), dtype=np.float32)
            self.hist_state_buffer = np.zeros((self.buffer_capacity, hist_state[1].shape[0], hist_state[1].shape[1]), dtype=np.float32)
            self.action_buffer = np.zeros((self.buffer_capacity, action.shape[0]), dtype=np.float32)
            self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
            self.next_state_buffer = np.zeros((self.buffer_capacity, next_state[0].shape[0]), dtype=np.float32)
            self.next_hist_state_buffer = np.zeros((self.buffer_capacity, next_hist_state[1].shape[0], next_hist_state[1].shape[1]), dtype=np.float32)
            self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
            self.old_prob_buffer = np.zeros((self.buffer_capacity, action.shape[0]), dtype=np.float32) # Initialize old_prob buffer
            self.value_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32) # Initialize value buffer


        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = state[0]
        self.hist_state_buffer[index] = hist_state[1]
        self.action_buffer[index] = action
        self.reward_buffer[index] = np.array([reward])
        self.next_state_buffer[index] = next_state[0]
        self.next_hist_state_buffer[index] = next_state[1]
        self.done_buffer[index] = np.array([float(done)])
        self.old_prob_buffer[index] = old_prob # Store log prob
        self.value_buffer[index] = value


        self.buffer_counter += 1

    def ppo_sample(self, batch_size):
        """Samples a batch for PPO training, and calculates advantages (GAE)."""
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices], dtype=tf.float32)
        hist_state_batch = tf.convert_to_tensor(self.hist_state_buffer[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices], dtype=tf.float32)
        next_hist_state_batch = tf.convert_to_tensor(self.next_hist_state_buffer[batch_indices], dtype=tf.float32)
        done_batch = tf.convert_to_tensor(self.done_buffer[batch_indices], dtype=tf.float32)
        old_prob_batch = tf.convert_to_tensor(self.old_prob_buffer[batch_indices], dtype=tf.float32)
        value_batch = tf.convert_to_tensor(self.value_buffer[batch_indices], dtype=tf.float32)


        # --- Calculate Advantages using GAE (Generalized Advantage Estimation) ---
        gamma = 0.99
        lamda = 0.95 # GAE lambda parameter
        deltas = reward_batch + gamma * (1.0 - done_batch) * self.value_buffer[batch_indices] - value_batch # TD errors
        advantage_batch = np.zeros_like(reward_batch)
        gae_advantage = 0
        for t in reversed(range(len(reward_batch))): # Iterate backwards in time
            gae_advantage = deltas[t] + gamma * lamda * (1 - done_batch[t]) * gae_advantage
            advantage_batch[t] = gae_advantage
        advantage_batch = (advantage_batch - np.mean(advantage_batch)) / (np.std(advantage_batch) + 1e-8) # Normalize advantages


        return state_batch, hist_state_batch, action_batch, old_prob_batch, advantage_batch, value_batch

    def clear_buffer(self): # Clear buffer at the start of each episode for on-policy
        self.buffer_counter = 0
        self.ep_start_index = 0


# --- 4. Training Loop ---
def evaluate_agent(agent, test_data, chart_save_dir, episode_num):
    """... (rest of the evaluate_agent function - no changes) ..."""
    test_env = CryptoTradingEnvironment(test_data, n_periods=730, sharpe_window=3000000, slippage_factor=0.0, stop_loss_pct=0.05) # Use test data and same env params
    state = test_env.reset()
    episode_portfolio_values = []
    episode_bitcoin_values = []

    for timestep in range(len(list(test_data.values())[0]) -1): # Evaluate over the entire test dataset
        action = agent.get_action(state, noise=False) # No exploration noise during evaluation

        action_set = {}
        action_set['cash'] = action[0]
        action_set['btc'] = action[1]
        action_set['eth'] = action[2]
        action_set['xrp'] = action[3]
        action_set['sol'] = action[4]

        action_set['btc_sl'] = action[5]
        action_set['eth_sl'] = action[6]
        action_set['xrp_sl'] = action[7]
        action_set['sol_sl'] = action[8]

        next_state, _, done, _ = test_env.step(action_set) # Reward is not important for evaluation

        episode_portfolio_values.append(test_env.portfolio_value_history[-1])
        episode_bitcoin_values.append(test_env.bitcoin_value_history[-1])

        state = next_state
        if done:
            break

    # --- Save Evaluation Chart ---
    plt.figure(figsize=(12, 6))
    plt.plot(episode_portfolio_values, label='PPO Agent Portfolio Value (Test Data)') # Changed label
    plt.plot(episode_bitcoin_values, label='Hold Bitcoin Portfolio Value (Test Data)', linestyle='--')
    plt.xlabel('Timestep')
    plt.ylabel('Portfolio Value')
    plt.title(f'Episode {episode_num} - Test Data Portfolio Value Comparison') # Updated chart title
    plt.legend()
    chart_path = os.path.join(chart_save_dir, f"test_episode_{episode_num}.png") # Changed filename
    plt.savefig(chart_path)
    plt.close() # Close plot to prevent display and clear memory
    print(f"Saved test data chart to {chart_path}")


if __name__ == '__main__':
    # --- Hyperparameters (Tune these!) ---
    episodes = 1000 	# Number of training episodes
    timesteps_per_episode = 730 # Length of each episode (730 hours in a month)
    batch_size = 64 	# Batch size for training updates
    buffer_capacity = 730 * 30 # Make buffer size episode length * some factor for PPO
    train_epochs_ppo = 10 # PPO training epochs per episode
    sharpe_window = 24 # Window size for Sharpe Ratio calculation
    slippage_factor = 0.01 # Slippage factor (e.g., 0.01 for 1% slippage) # No slippage for now to check convergence
    stop_loss_pct = 0.05 # Stop loss percentage (e.g., 0.05 for 5% drop)
    model_save_interval = 10 # Save model every n episodes

    # --- Directories for saving models and charts ---
    model_save_dir = "saved_models" # Changed directory name for PPO
    chart_save_dir = "episode_charts" # Changed directory name for PPO
    os.makedirs(model_save_dir, exist_ok=True) # Create directory if it doesn't exist
    os.makedirs(chart_save_dir, exist_ok=True) # Create directory if it doesn't exist


    # --- Get Data ---
    crypto_data = get_crypto_data()
    train_data, val_data = split_data(crypto_data)
    test_data = val_data # Use val_data as test_data as per the original request

    # --- Environment and Agent Setup ---
    env = CryptoTradingEnvironment(train_data, n_periods=730, sharpe_window=sharpe_window, slippage_factor=slippage_factor, stop_loss_pct=stop_loss_pct, random_start=True)
    current_state_dim = (9,)
    hist_state_dim = (730, 17*len(env.coins),)
    action_dim = len(env.coins) * 2 + 1 # *2 for Stop Loss actions, +1 for cash
    agent = PPOAgent(current_state_dim, hist_state_dim, action_dim) # Use PPOAgent
    replay_buffer = ReplayBuffer(buffer_capacity) # Replay buffer for PPO


    # --- Training Loop ---
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = []
        episode_portfolio_values = []
        episode_bitcoin_values = []
        replay_buffer.clear_buffer() # Clear buffer at start of each episode for PPO

        for timestep in range(timesteps_per_episode):
            action, old_policy_prob = agent.get_action(state, noise=True, old_policy=True) # Get action and old policy prob

            action_set = {}
            action_set['cash'] = action[0]
            action_set['btc'] = action[1]
            action_set['eth'] = action[2]
            action_set['xrp'] = action[3]
            action_set['sol'] = action[4]

            action_set['btc_sl'] = action[5]
            action_set['eth_sl'] = action[6]
            action_set['xrp_sl'] = action[7]
            action_set['sol_sl'] = action[8]

            value = agent.critic_value(state).numpy()[0][0] # Get state value from critic

            next_state, reward, done, _ = env.step(action_set)

            episode_rewards.append(reward)
            replay_buffer.record(state, state, action, reward, next_state, next_state, done, old_policy_prob, value) # Record with old_prob and value
            episode_portfolio_values.append(env.portfolio_value_history[-1]) # Record portfolio value
            episode_bitcoin_values.append(env.bitcoin_value_history[-1]) # Record bitcoin value

            state = next_state
            if done:
                break

        # --- PPO Training Epochs ---
        if replay_buffer.buffer_counter > batch_size: # Train only if enough data is in buffer
            for _ in range(agent.train_epochs): # Multiple epochs of training per episode for PPO
                actor_loss, critic_loss = agent.train_step(replay_buffer, batch_size)
            print(f"Episode {episode+1}/{episodes}, Actor Loss: {actor_loss.numpy():.4e}, Critic Loss: {critic_loss.numpy():.4e}")


        avg_reward = np.mean(episode_rewards)
        print(f"Episode {episode+1}/{episodes}, Average Reward: {avg_reward:.4f}, Portfolio Value: {env.get_current_portfolio_value():.2f}")

        # --- Save Model Snapshot ---
        if (episode + 1) % model_save_interval == 0:
            agent.actor_model.save(os.path.join(model_save_dir, f"actor_episode_{episode+1}.keras"))
            agent.critic_model.save(os.path.join(model_save_dir, f"critic_episode_{episode+1}.keras")) # Save critic as well

        if (episode + 1) % 10 == 0:
            evaluate_agent(agent, test_data, chart_save_dir, episode + 1)